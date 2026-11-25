import ast
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external"

ASPECT_FIX = {"prices": "price", "Food": "food", "Service": "service"}


def _read_two_col(path):
    sents, tags, s, t = [], [], [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            if s:
                sents.append(s)
                tags.append(t)
            s, t = [], []
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        s.append(parts[0])
        t.append(parts[-1])
    if s:
        sents.append(s)
        tags.append(t)
    return sents, tags


def load_hinglid(split="train", root=EXT):
    return _read_two_col(Path(root) / "code-mixed-nlp" / "L3Cube-HingLID" / f"{split}.txt")


def load_conll(path):
    """Generic token<TAB>...<TAB>tag file, blank line between sentences (LinCE / POS)."""
    return _read_two_col(path)


def load_ner(root=EXT):
    df = pd.read_csv(Path(root) / "Named-Entity-Recognition" / "Twitterdata" / "annotatedData.csv")
    sents, tags, s, t, cur = [], [], [], [], None
    for sid, w, tag in zip(df["Sent"], df["Word"], df["Tag"]):
        if not isinstance(sid, str) or sid != cur:
            if s:
                sents.append(s)
                tags.append(t)
            s, t = [], []
            cur = sid if isinstance(sid, str) else None
        if isinstance(sid, str) and isinstance(w, str):
            s.append(w)
            t.append("O" if tag == "Other" else tag)
    if s:
        sents.append(s)
        tags.append(t)
    return sents, tags


def split_data(xs, ys, ratios=(0.7, 0.15, 0.15), seed=42):
    idx = list(range(len(xs)))
    random.Random(seed).shuffle(idx)
    a = int(ratios[0] * len(idx))
    b = a + int(ratios[1] * len(idx))
    pick = lambda ids: ([xs[i] for i in ids], [ys[i] for i in ids])
    return pick(idx[:a]), pick(idx[a:b]), pick(idx[b:])


def load_acsa(split="train", root=EXT):
    df = pd.read_csv(Path(root) / "ACSA-Mix" / "data" / f"{split}.csv")
    rows, skipped = [], 0
    for _, r in df.iterrows():
        try:
            cats = ast.literal_eval(r["aspect_category_now"])
            pols = ast.literal_eval(r["ac_polarity"])
        except (ValueError, SyntaxError):
            skipped += 1
            continue
        if len(cats) != len(pols) or not isinstance(r["text"], str):
            skipped += 1
            continue
        pairs, seen = [], set()
        for c, p in zip(cats, pols):
            c = ASPECT_FIX.get(c, c).lower()
            if c not in seen:
                pairs.append((c, p.lower()))
                seen.add(c)
        rows.append({"id": str(r["id"]), "text": r["text"], "aspects": pairs})
    return rows, skipped


def load_sentimix(path):
    """SemEval-2020 Task 9 CoNLL format: 'meta<TAB>id<TAB>label' then token<TAB>lang lines."""
    texts, labels, langs, cur, cur_l, label = [], [], [], [], [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if parts[0] == "meta":
            if cur:
                texts.append(cur)
                labels.append(label)
                langs.append(cur_l)
            cur, cur_l = [], []
            label = parts[2].strip() if len(parts) > 2 else None
        elif len(parts) >= 2 and parts[0]:
            cur.append(parts[0])
            cur_l.append(parts[1])
    if cur:
        texts.append(cur)
        labels.append(label)
        langs.append(cur_l)
    return texts, labels, langs


def load_own(path):
    """Own annotated CSV: review_id,text,aspect,sentiment (one row per aspect; aspect='none' if none)."""
    df = pd.read_csv(path)
    grouped = defaultdict(lambda: {"text": None, "aspects": []})
    for _, r in df.iterrows():
        g = grouped[str(r["review_id"])]
        g["text"] = r["text"]
        if str(r["aspect"]).lower() not in ("none", "nan", ""):
            g["aspects"].append((str(r["aspect"]).lower(), str(r["sentiment"]).lower()))
    return [{"id": k, **v} for k, v in grouped.items()]
