"""Merge classical (results/results.json) and transformer (results/transformers/*.json)
scores into one table: results/comparison.md"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
NAMES = {
    "bert-base-multilingual-cased": "mBERT",
    "l3cube-pune/hing-bert": "HingBERT",
    "l3cube-pune/hing-mbert-mixed": "HingMBERT-mixed",
    "l3cube-pune/hing-roberta-mixed": "HingRoBERTa-mixed",
    "google/muril-base-cased": "MuRIL",
    "xlm-roberta-base": "XLM-R",
}
METRIC = {"lid": ("macro_f1", "accuracy"), "ner": ("f1", "precision"), "acd": ("f1", "precision"), "acsc": ("macro_f1", "accuracy")}
TITLES = {"lid": "Language ID (HingLID test)", "ner": "NER (Singh et al. 2018, same split)", "acd": "Aspect category detection (ACSA-Mix)", "acsc": "Aspect sentiment (ACSA-Mix)"}


def classical_rows(res, task):
    if task == "lid":
        return [(r["model"], r["macro_f1"], r["accuracy"]) for r in res.get("lid", [])]
    if task == "ner":
        return [(r["model"], r["f1"], r["precision"]) for r in res.get("ner", [])]
    rows = [r for r in res.get("absa", []) if r["script"] == "romanized"]
    if task == "acd":
        return [(r["model"], r["acd_f1"], r["acd_precision"]) for r in rows]
    return [(r["model"], r["acsc_macro_f1"], r["acsc_accuracy"]) for r in rows]


def main():
    res = json.loads((RES / "results.json").read_text()) if (RES / "results.json").exists() else {}
    trans = [json.loads(p.read_text()) for p in sorted((RES / "transformers").glob("*.json"))] if (RES / "transformers").exists() else []
    L = ["# Classical vs transformer", ""]
    for task in ("lid", "ner", "acd", "acsc"):
        main_m, second = METRIC[task]
        rows = [("classical", m, a, b) for m, a, b in classical_rows(res, task)]
        rows += [("transformer", NAMES.get(t["model"], t["model"]), t["test"][main_m], t["test"].get(second, 0.0)) for t in trans if t["task"] == task]
        if not rows:
            continue
        best = max(r[2] for r in rows)
        L += [f"## {TITLES[task]}", "", f"| Type | Model | {main_m} | {second} |", "|---|---|---|---|"]
        for kind, m, a, b in sorted(rows, key=lambda r: -r[2]):
            mark = "**" if a == best else ""
            L.append(f"| {kind} | {mark}{m}{mark} | {mark}{100 * a:.2f}{mark} | {100 * b:.2f} |")
        L.append("")
    (RES / "comparison.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
