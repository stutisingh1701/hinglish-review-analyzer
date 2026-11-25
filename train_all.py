"""Train every classical model and write results/results.{md,json}.

    python scripts/train_all.py                 # all real-data experiments
    python scripts/train_all.py --pos-train data/pos/train.conll --pos-test data/pos/test.conll
    python scripts/train_all.py --sentimix-train train.txt --sentimix-test test.txt
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hinglish.absa import AspectCategoryDetector, AspectSentimentClassifier, make_classifier, text_features  # noqa: E402
from hinglish.crf_tagger import CRFTagger  # noqa: E402
from hinglish.data import load_acsa, load_conll, load_hinglid, load_ner, load_sentimix, split_data  # noqa: E402
from hinglish.hmm import HMMTagger  # noqa: E402
from hinglish.langid import CRFLID, NgramLID  # noqa: E402
from hinglish.metrics import classification_scores, multilabel_scores, span_scores, token_scores  # noqa: E402
from hinglish.normalize import normalize  # noqa: E402

MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def timed(fn):
    t = time.time()
    out = fn()
    return out, round(time.time() - t, 1)


def log(msg):
    print(msg, flush=True)


def run_lid(res, max_crf):
    log("== Language identification (L3Cube-HingLID)")
    tr, va, te = load_hinglid("train"), load_hinglid("validation"), load_hinglid("test")
    res["lid_data"] = {s: {"sentences": len(d[0]), "tokens": sum(map(len, d[0]))} for s, d in (("train", tr), ("validation", va), ("test", te))}
    rows = []
    ngram, t1 = timed(lambda: NgramLID(n=4, k=0.5).fit(*tr))
    hmm, t2 = timed(lambda: HMMTagger(k=0.1).fit(*tr))
    crf_tr = (tr[0][:max_crf], tr[1][:max_crf]) if max_crf else tr
    crf, t3 = timed(lambda: CRFLID(None, max_iterations=60).fit(*crf_tr))
    crf_lm, t4 = timed(lambda: CRFLID(ngram, max_iterations=60).fit(*crf_tr))
    preds = {}
    for name, model, secs in (("Char 4-gram LM (add-k, no context)", ngram, t1), ("HMM bigram + Viterbi", hmm, t2),
                              ("CRF (word/char features)", crf, t3), ("CRF + char-LM features", crf_lm, t4)):
        p = model.predict(te[0])
        preds[name] = p
        s = token_scores(te[1], p)
        rows.append({"model": name, "accuracy": s["accuracy"], "macro_f1": s["macro_f1"], "train_sec": secs})
        log(f"   {name:38s} acc={s['accuracy']:.4f} macroF1={s['macro_f1']:.4f} ({secs}s)")
    res["lid"] = rows
    res["lid_crf_train_sentences"] = len(crf_tr[0])
    joblib.dump(ngram, MODELS / "lid_ngram.joblib")
    joblib.dump(hmm, MODELS / "lid_hmm.joblib")
    crf_rows = {r["model"]: r["macro_f1"] for r in rows}
    best_crf = crf_lm if crf_rows["CRF + char-LM features"] >= crf_rows["CRF (word/char features)"] else crf
    joblib.dump(best_crf, MODELS / "lid_crf.joblib")

    best = max(rows, key=lambda r: r["macro_f1"])["model"]
    hmm_p, best_p = preds["HMM bigram + Viterbi"], preds[best]
    err = Counter()
    for s, g, a, b in zip(te[0], te[1], hmm_p, best_p):
        for w, gg, aa, bb in zip(s, g, a, b):
            if aa != gg:
                err[(w.lower(), gg, aa, "fixed" if bb == gg else "still wrong")] += 1
    pd.DataFrame([{"word": k[0], "gold": k[1], "hmm_pred": k[2], f"{best}": k[3], "count": v} for k, v in err.most_common(200)]).to_csv(RESULTS / "errors_lid_hmm.csv", index=False)
    return ngram


def run_ner(res, ngram):
    log("== Named entity recognition (Singh et al. 2018 Hinglish tweets)")
    sents, tags = load_ner()
    tr, va, te = split_data(sents, tags, seed=42)
    tr = (tr[0] + va[0], tr[1] + va[1])
    res["ner_data"] = {"sentences": len(sents), "train": len(tr[0]), "test": len(te[0]),
                       "entities": dict(Counter(t[2:] for seq in tags for t in seq if t.startswith("B-")))}
    lid_feats = lambda xs: [ngram.features(s) for s in xs]
    hmm, t1 = timed(lambda: HMMTagger(k=0.1).fit(*tr))
    crf, t2 = timed(lambda: CRFTagger(max_iterations=150).fit(*tr))
    crf_lid, t3 = timed(lambda: CRFTagger(max_iterations=150).fit(tr[0], tr[1], lid_feats(tr[0])))
    rows = []
    for name, pred, secs in (
        ("HMM bigram + Viterbi", hmm.predict(te[0]), t1),
        ("CRF (word/char/context)", crf.predict(te[0]), t2),
        ("CRF + language-ID features", crf_lid.predict(te[0], lid_feats(te[0])), t3),
    ):
        s = span_scores(te[1], pred)
        row = {"model": name, "precision": s["micro"]["precision"], "recall": s["micro"]["recall"], "f1": s["micro"]["f1"], "train_sec": secs}
        for lab in ("Per", "Org", "Loc"):
            row[f"f1_{lab}"] = s.get(lab, {}).get("f1", 0.0)
        rows.append(row)
        log(f"   {name:38s} P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f}")
    res["ner"] = rows
    best_lid = rows[2]["f1"] >= rows[1]["f1"]
    joblib.dump(hmm, MODELS / "ner_hmm.joblib")
    joblib.dump(crf_lid if best_lid else crf, MODELS / "ner_crf.joblib")
    flag = MODELS / "ner_crf.uses_lid"
    if best_lid:
        flag.write_text("1")
    elif flag.exists():
        flag.unlink()


def run_pos(res, train_path, test_path):
    log("== POS tagging (user-supplied CoNLL)")
    tr, te = load_conll(train_path), load_conll(test_path)
    hmm, t1 = timed(lambda: HMMTagger(k=0.1).fit(*tr))
    crf, t2 = timed(lambda: CRFTagger(max_iterations=150).fit(*tr))
    rows = []
    for name, model, secs in (("HMM bigram + Viterbi", hmm, t1), ("CRF", crf, t2)):
        s = token_scores(te[1], model.predict(te[0]))
        rows.append({"model": name, "accuracy": s["accuracy"], "macro_f1": s["macro_f1"], "train_sec": secs})
        log(f"   {name:38s} acc={s['accuracy']:.4f} macroF1={s['macro_f1']:.4f}")
    res["pos"] = rows
    joblib.dump(hmm, MODELS / "pos_hmm.joblib")
    joblib.dump(crf, MODELS / "pos_crf.joblib")


def run_absa(res):
    log("== Aspect category detection + aspect sentiment (ACSA-Mix)")
    tr, skip_tr = load_acsa("train")
    te, skip_te = load_acsa("test")
    res["acsa_data"] = {"train": len(tr), "test": len(te), "skipped_train": skip_tr, "skipped_test": skip_te,
                        "aspects_train": dict(Counter(a for r in tr for a, _ in r["aspects"])),
                        "polarity_train": dict(Counter(p for r in tr for _, p in r["aspects"]))}
    rows = []
    for script, roman in (("romanized", True), ("original script", False)):
        Xtr = [normalize(r["text"], romanize=roman) for r in tr]
        Xte = [normalize(r["text"], romanize=roman) for r in te]
        gold_sets = [[a for a, _ in r["aspects"]] for r in te]
        ptr = [(x, a, p) for x, r in zip(Xtr, tr) for a, p in r["aspects"]]
        pte = [(x, a, p) for x, r in zip(Xte, te) for a, p in r["aspects"]]
        gold_pairs = [set(r["aspects"]) for r in te]
        for clf in ("nb", "lr", "svm"):
            acd = AspectCategoryDetector(clf).fit(Xtr, [[a for a, _ in r["aspects"]] for r in tr])
            pred_sets = acd.predict(Xte)
            acd_s = multilabel_scores(gold_sets, pred_sets)
            acsc = AspectSentimentClassifier(clf).fit([x for x, _, _ in ptr], [a for _, a, _ in ptr], [p for _, _, p in ptr])
            pol = acsc.predict([x for x, _, _ in pte], [a for _, a, _ in pte])
            sc = classification_scores([p for _, _, p in pte], pol)
            joint_pred = []
            for x, cats in zip(Xte, pred_sets):
                labs = acsc.predict([x] * len(cats), cats)
                joint_pred.append(set(zip(cats, labs)))
            joint = multilabel_scores(gold_pairs, joint_pred)
            rows.append({"script": script, "model": {"nb": "Complement NB", "lr": "Logistic Regression", "svm": "Linear SVM"}[clf],
                         "acd_precision": acd_s["precision"], "acd_recall": acd_s["recall"], "acd_f1": acd_s["f1"],
                         "acsc_accuracy": sc["accuracy"], "acsc_macro_f1": sc["macro_f1"], "joint_f1": joint["f1"]})
            log(f"   {script:16s} {clf:4s} ACD F1={acd_s['f1']:.4f}  ACSC acc={sc['accuracy']:.4f} mF1={sc['macro_f1']:.4f}  joint F1={joint['f1']:.4f}")
            if roman:
                joblib.dump(acd, MODELS / f"acd_{clf}.joblib")
                joblib.dump(acsc, MODELS / f"acsc_{clf}.joblib")
                if clf == "lr":
                    res["acsc_report_lr"] = sc["report"]
                    pd.DataFrame([{"text": x, "aspect": a, "gold": p, "pred": q} for (x, a, p), q in zip(pte, pol) if p != q]).to_csv(RESULTS / "errors_acsc_lr.csv", index=False)
    res["absa"] = rows


def run_sentimix(res, train_path, test_path):
    log("== Sentence sentiment (SemEval-2020 SentiMix)")
    from sklearn.pipeline import make_pipeline
    tr_x, tr_y, _ = load_sentimix(train_path)
    te_x, te_y, _ = load_sentimix(test_path)
    Xtr = [normalize(" ".join(t)) for t in tr_x]
    Xte = [normalize(" ".join(t)) for t in te_x]
    rows = []
    for clf in ("nb", "lr", "svm"):
        m = make_pipeline(text_features(), make_classifier(clf)).fit(Xtr, tr_y)
        s = classification_scores(te_y, m.predict(Xte))
        rows.append({"model": clf, "accuracy": s["accuracy"], "macro_f1": s["macro_f1"]})
        log(f"   {clf:4s} acc={s['accuracy']:.4f} macroF1={s['macro_f1']:.4f}")
        joblib.dump(m, MODELS / f"sentimix_{clf}.joblib")
    res["sentimix"] = rows


def fmt(x):
    return f"{100 * x:.2f}" if isinstance(x, float) else str(x)


def write_markdown(res):
    L = ["# Results (classical models)", "", f"Generated by `scripts/train_all.py` on {time.strftime('%Y-%m-%d')}. All scores in %.", ""]
    if "lid" in res:
        d = res["lid_data"]
        L += ["## 1. Word-level language identification", "",
              f"Data: L3Cube-HingLID, train {d['train']['sentences']} sentences / {d['train']['tokens']} tokens, test {d['test']['sentences']} / {d['test']['tokens']}. Tags: HI, EN. CRFs trained on {res['lid_crf_train_sentences']} sentences.", "",
              "| Model | Accuracy | Macro-F1 | Train time (s) |", "|---|---|---|---|"]
        L += [f"| {r['model']} | {fmt(r['accuracy'])} | {fmt(r['macro_f1'])} | {r['train_sec']} |" for r in res["lid"]]
        L += ["", "Most frequent HMM errors: `results/errors_lid_hmm.csv`.", ""]
    if "ner" in res:
        d = res["ner_data"]
        L += ["## 2. Named entity recognition", "",
              f"Data: Singh et al. (2018) Hinglish NER tweets, {d['sentences']} sentences, random 85/15 split (seed 42). Entities (B- tags): {d['entities']}. Exact-span micro scores.", "",
              "| Model | Precision | Recall | F1 | F1 Person | F1 Org | F1 Location |", "|---|---|---|---|---|---|---|"]
        L += [f"| {r['model']} | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | {fmt(r['f1_Per'])} | {fmt(r['f1_Org'])} | {fmt(r['f1_Loc'])} |" for r in res["ner"]]
        L += [""]
    if "pos" in res:
        L += ["## POS tagging", "", "| Model | Accuracy | Macro-F1 |", "|---|---|---|"]
        L += [f"| {r['model']} | {fmt(r['accuracy'])} | {fmt(r['macro_f1'])} |" for r in res["pos"]]
        L += [""]
    if "absa" in res:
        d = res["acsa_data"]
        L += ["## 3. Aspect category detection (ACD) and aspect sentiment (ACSC)", "",
              f"Data: ACSA-Mix (restaurant reviews, Hindi-English), train {d['train']} / test {d['test']} sentences ({d['skipped_test']} test rows skipped for missing labels). Aspects: {d['aspects_train']}. Polarity: {d['polarity_train']}.", "",
              "ACD = micro-F1 over aspect categories. ACSC = sentiment given gold aspect. Joint = (aspect, polarity) pairs from the full pipeline.", "",
              "| Text | Model | ACD P | ACD R | ACD F1 | ACSC Acc | ACSC Macro-F1 | Joint F1 |", "|---|---|---|---|---|---|---|---|"]
        L += [f"| {r['script']} | {r['model']} | {fmt(r['acd_precision'])} | {fmt(r['acd_recall'])} | {fmt(r['acd_f1'])} | {fmt(r['acsc_accuracy'])} | {fmt(r['acsc_macro_f1'])} | {fmt(r['joint_f1'])} |" for r in res["absa"]]
        if "acsc_report_lr" in res:
            L += ["", "Per-class ACSC report (Logistic Regression, romanized):", "", "```", res["acsc_report_lr"].rstrip(), "```"]
        L += [""]
    if "sentimix" in res:
        L += ["## Sentence sentiment (SentiMix)", "", "| Model | Accuracy | Macro-F1 |", "|---|---|---|"]
        L += [f"| {r['model']} | {fmt(r['accuracy'])} | {fmt(r['macro_f1'])} |" for r in res["sentimix"]]
        L += [""]
    (RESULTS / "results.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lid-crf-sentences", type=int, default=0, help="limit CRF LID training sentences (0 = all)")
    ap.add_argument("--only", nargs="*", choices=["lid", "ner", "absa", "pos", "sentimix"])
    ap.add_argument("--pos-train")
    ap.add_argument("--pos-test")
    ap.add_argument("--sentimix-train")
    ap.add_argument("--sentimix-test")
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    res_path = RESULTS / "results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() and args.only else {}
    todo = set(args.only or ["lid", "ner", "absa"])
    if args.pos_train:
        todo.add("pos")
    if args.sentimix_train:
        todo.add("sentimix")
    ngram = None
    if "lid" in todo:
        ngram = run_lid(res, args.lid_crf_sentences)
    if "ner" in todo:
        if ngram is None:
            ngram = joblib.load(MODELS / "lid_ngram.joblib")
        run_ner(res, ngram)
    if "absa" in todo:
        run_absa(res)
    if "pos" in todo:
        run_pos(res, args.pos_train, args.pos_test)
    if "sentimix" in todo:
        run_sentimix(res, args.sentimix_train, args.sentimix_test)
    res_path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    write_markdown(res)
    log(f"Saved models to {MODELS} and results to {RESULTS / 'results.md'}")


if __name__ == "__main__":
    main()
