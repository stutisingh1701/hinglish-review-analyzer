"""Convert a Label Studio JSON export (annotation/label_studio_config.xml) to
review_id,text,aspect,sentiment rows. Writes one CSV per annotator.

    python scripts/labelstudio_to_csv.py export.json --out-dir data/own/
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ASPECTS = ["battery", "camera", "display", "performance", "price", "design", "audio", "service"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--out-dir", default="data/own")
    args = ap.parse_args()
    tasks = json.loads(Path(args.export).read_text(encoding="utf-8"))
    per_annotator = defaultdict(list)
    for task in tasks:
        text = task["data"]["text"]
        rid = task["data"].get("review_id", task["id"])
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            who = str(ann.get("completed_by", "a"))
            choice = {r["from_name"]: r["value"]["choices"][0] for r in ann["result"] if r.get("type") == "choices"}
            if choice.get("usable", "usable") != "usable":
                continue
            rows = [{"review_id": rid, "text": text, "aspect": a, "sentiment": choice[a]}
                    for a in ASPECTS if choice.get(a, "not mentioned") != "not mentioned"]
            per_annotator[who].extend(rows or [{"review_id": rid, "text": text, "aspect": "none", "sentiment": "none"}])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for who, rows in per_annotator.items():
        path = out / f"annotated_{who}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"annotator {who}: {len({r['review_id'] for r in rows})} reviews -> {path}")


if __name__ == "__main__":
    main()
