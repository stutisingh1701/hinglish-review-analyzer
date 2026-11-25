"""Keep only genuinely code-mixed Hinglish comments, using the trained LID model
and the Code-Mixing Index, and pre-tag likely aspects to speed up annotation.

    python scripts/filter_hinglish.py data/own/raw_comments.csv --keep 500
Output: data/own/to_annotate.csv
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hinglish.absa import ProductAspectExtractor  # noqa: E402
from hinglish.pipeline import HinglishAnalyzer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--min-tokens", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--min-cmi", type=float, default=10.0)
    ap.add_argument("--min-each", type=int, default=1, help="min tokens of each language")
    ap.add_argument("--keep", type=int, default=500)
    ap.add_argument("--require-aspect", action="store_true", help="keep only comments mentioning a product aspect")
    ap.add_argument("--out", default=str(ROOT / "data" / "own" / "to_annotate.csv"))
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    an = HinglishAnalyzer()
    ext = ProductAspectExtractor()
    rows = []
    for _, r in df.iterrows():
        text = str(r["text"])
        if re.search(r"[ऀ-ॿ]", text) and not re.search(r"[a-zA-Z]{3,}", text):
            continue
        out = an.analyze(text)
        langs = [l for l in out["lang"] if l != "OTHER"]
        if not (args.min_tokens <= len(langs) <= args.max_tokens):
            continue
        if langs.count("HI") < args.min_each or langs.count("EN") < args.min_each:
            continue
        if (out["cmi"] or 0) < args.min_cmi:
            continue
        aspects = sorted({m["aspect"] for m in ext.find(out["tokens"])})
        if args.require_aspect and not aspects:
            continue
        rows.append({"review_id": r.get("comment_id", len(rows)), "text": text, "cmi": out["cmi"],
                     "hi_tokens": langs.count("HI"), "en_tokens": langs.count("EN"), "suggested_aspects": ",".join(aspects)})
    kept = pd.DataFrame(rows)
    if kept.empty:
        sys.exit("Nothing passed the filters; relax --min-cmi / --min-each.")
    kept = kept.sort_values("cmi", ascending=False).head(args.keep).sample(frac=1, random_state=42)
    kept.to_csv(args.out, index=False)
    print(f"{len(df)} comments -> {len(rows)} code-mixed -> kept {len(kept)} in {args.out}")
    print(f"mean CMI {kept['cmi'].mean():.1f}; with suggested aspect: {(kept['suggested_aspects'] != '').mean():.0%}")


if __name__ == "__main__":
    main()
