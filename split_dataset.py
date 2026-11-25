"""Split an annotated CSV into train/val/test by review (70/15/15).

    python scripts/split_dataset.py data/own/annotated.csv
"""
import argparse
import random
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    ids = sorted(df["review_id"].astype(str).unique())
    random.Random(args.seed).shuffle(ids)
    a, b = int(0.7 * len(ids)), int(0.85 * len(ids))
    parts = {"train": set(ids[:a]), "val": set(ids[a:b]), "test": set(ids[b:])}
    src = Path(args.csv)
    for name, keep in parts.items():
        out = src.with_name(f"{src.stem}_{name}.csv")
        df[df["review_id"].astype(str).isin(keep)].to_csv(out, index=False)
        print(f"{name}: {len(keep)} reviews -> {out}")


if __name__ == "__main__":
    main()
