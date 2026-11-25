"""Evaluate the product-review pipeline on your own annotated CSV
(review_id,text,aspect,sentiment — one row per aspect).

    python scripts/evaluate_own.py data/own/annotated.csv
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hinglish.data import load_own  # noqa: E402
from hinglish.metrics import classification_scores, multilabel_scores  # noqa: E402
from hinglish.pipeline import HinglishAnalyzer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=str(ROOT / "data" / "own" / "annotated.csv"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = load_own(args.csv)
    an = HinglishAnalyzer()
    gold_asp, pred_asp, gold_pairs, pred_pairs, pol_gold, pol_pred, mistakes = [], [], [], [], [], [], []
    for r in rows:
        out = an.analyze(r["text"])
        pred = {(a["aspect"], a["label"]) for a in out["aspects"]}
        gold = set(r["aspects"])
        gold_asp.append({a for a, _ in gold})
        pred_asp.append({a for a, _ in pred})
        gold_pairs.append(gold)
        pred_pairs.append(pred)
        pred_by_aspect = {}
        for a, l in pred:
            pred_by_aspect.setdefault(a, set()).add(l)
        for a, l in gold:
            if a in pred_by_aspect:
                pol_gold.append(l)
                pol_pred.append(l if l in pred_by_aspect[a] else sorted(pred_by_aspect[a])[0])
        if gold != pred:
            mistakes.append({"text": r["text"], "gold": sorted(gold), "pred": sorted(pred)})
    asp = multilabel_scores(gold_asp, pred_asp)
    joint = multilabel_scores(gold_pairs, pred_pairs)
    pol = classification_scores(pol_gold, pol_pred) if pol_gold else {"accuracy": 0.0, "macro_f1": 0.0}
    res = {"reviews": len(rows), "aspect_f1": asp["f1"], "aspect_precision": asp["precision"], "aspect_recall": asp["recall"],
           "sentiment_accuracy_on_found_aspects": pol["accuracy"], "sentiment_macro_f1": pol["macro_f1"], "joint_f1": joint["f1"]}
    print(json.dumps(res, indent=2))
    print(f"\n{len(mistakes)} reviews with any mismatch; first 10:")
    for m in mistakes[:10]:
        print(f"  {m['text']}\n     gold={m['gold']}  pred={m['pred']}")
    if args.out:
        Path(args.out).write_text(json.dumps({"scores": res, "mistakes": mistakes}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
