"""Inter-annotator agreement (Cohen's kappa) between two annotation CSVs.

    python scripts/kappa.py data/own/annotated_1.csv data/own/annotated_2.csv
Reports kappa for (a) aspect presence and (b) sentiment on aspects both marked.
"""
import argparse
from collections import defaultdict

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ASPECTS = ["battery", "camera", "display", "performance", "price", "design", "audio", "service"]


def load(path):
    df = pd.read_csv(path)
    d = defaultdict(dict)
    for _, r in df.iterrows():
        if str(r["aspect"]).lower() != "none":
            d[str(r["review_id"])][str(r["aspect"]).lower()] = str(r["sentiment"]).lower()
        else:
            d[str(r["review_id"])]
    return d


def interpret(k):
    return "poor" if k <= 0.2 else "fair" if k <= 0.4 else "moderate" if k <= 0.6 else "substantial" if k <= 0.8 else "almost perfect"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    common = sorted(set(A) & set(B))
    pres_a, pres_b, sent_a, sent_b = [], [], [], []
    for rid in common:
        for asp in ASPECTS:
            pres_a.append(asp in A[rid])
            pres_b.append(asp in B[rid])
            if asp in A[rid] and asp in B[rid]:
                sent_a.append(A[rid][asp])
                sent_b.append(B[rid][asp])
    k1 = cohen_kappa_score(pres_a, pres_b)
    print(f"Reviews annotated by both: {len(common)}")
    print(f"Aspect presence kappa : {k1:.3f} ({interpret(k1)})")
    if sent_a:
        k2 = cohen_kappa_score(sent_a, sent_b)
        agree = sum(x == y for x, y in zip(sent_a, sent_b)) / len(sent_a)
        print(f"Sentiment kappa       : {k2:.3f} ({interpret(k2)}), raw agreement {agree:.1%} on {len(sent_a)} shared aspects")


if __name__ == "__main__":
    main()
