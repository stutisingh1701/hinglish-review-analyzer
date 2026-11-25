from collections import Counter

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support


def flatten(seqs):
    return [x for s in seqs for x in s]


def token_scores(gold, pred):
    g, p = flatten(gold), flatten(pred)
    return {
        "accuracy": accuracy_score(g, p),
        "macro_f1": f1_score(g, p, average="macro"),
        "weighted_f1": f1_score(g, p, average="weighted"),
        "report": classification_report(g, p, digits=4, zero_division=0),
    }


def bio_spans(seq):
    spans, start, label = [], None, None
    for i, tag in enumerate(list(seq) + ["O"]):
        tag = "O" if tag in ("O", "Other") else tag
        if tag.startswith("B-") or tag == "O" or (tag.startswith("I-") and tag[2:] != label):
            if label is not None:
                spans.append((start, i, label))
                label = None
            if tag.startswith("B-") or tag.startswith("I-"):
                start, label = i, tag[2:]
    return spans


def span_scores(gold, pred):
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred):
        gs, ps = set(bio_spans(g)), set(bio_spans(p))
        for s in ps & gs:
            tp[s[2]] += 1
        for s in ps - gs:
            fp[s[2]] += 1
        for s in gs - ps:
            fn[s[2]] += 1
    out = {}
    for lab in sorted(set(tp) | set(fp) | set(fn)):
        out[lab] = _prf(tp[lab], fp[lab], fn[lab])
    out["micro"] = _prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    labs = [l for l in out if l != "micro"]
    out["macro_f1"] = sum(out[l]["f1"] for l in labs) / max(len(labs), 1)
    return out


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "support": tp + fn}


def multilabel_scores(gold_sets, pred_sets):
    tp = fp = fn = 0
    for g, p in zip(gold_sets, pred_sets):
        g, p = set(g), set(p)
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    return _prf(tp, fp, fn)


def classification_scores(gold, pred):
    p, r, f, _ = precision_recall_fscore_support(gold, pred, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f,
        "macro_precision": p,
        "macro_recall": r,
        "report": classification_report(gold, pred, digits=4, zero_division=0),
    }


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _lcs(a, b):
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge(reference, candidate):
    ref, cand = reference.lower().split(), candidate.lower().split()
    out = {}
    for n in (1, 2):
        r, c = _ngrams(ref, n), _ngrams(cand, n)
        overlap = sum((r & c).values())
        out[f"rouge{n}"] = _prf(overlap, sum(c.values()) - overlap, sum(r.values()) - overlap)["f1"]
    l = _lcs(ref, cand)
    p = l / len(cand) if cand else 0.0
    rr = l / len(ref) if ref else 0.0
    out["rougeL"] = 2 * p * rr / (p + rr) if p + rr else 0.0
    return out
