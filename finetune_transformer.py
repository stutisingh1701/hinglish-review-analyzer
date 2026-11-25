"""Fine-tune a transformer (HingBERT / mBERT / MuRIL ...) on the same splits as the
classical models. Run on a GPU (Colab / Kaggle).

    python scripts/finetune_transformer.py --task acsc --model l3cube-pune/hing-mbert-mixed
    python scripts/finetune_transformer.py --task lid  --model l3cube-pune/hing-bert --epochs 2
Tasks: lid (word language ID), ner, acd (aspect category, multi-label), acsc (aspect sentiment).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hinglish.data import load_acsa, load_hinglid, load_ner, split_data  # noqa: E402
from hinglish.metrics import classification_scores, multilabel_scores, span_scores, token_scores  # noqa: E402
from hinglish.normalize import normalize  # noqa: E402


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def limit(d, n):
    return d if not n else (d[0][:n], d[1][:n])


def token_data(task, n):
    if task == "lid":
        tr, va, te = load_hinglid("train"), load_hinglid("validation"), load_hinglid("test")
    else:
        sents, tags = load_ner()
        tr, va, te = split_data(sents, tags, seed=42)
    return limit(tr, n), limit(va, n), limit(te, n)


def acsa_data(roman, n):
    rows, _ = load_acsa("train")
    test, _ = load_acsa("test")
    random.Random(42).shuffle(rows)
    cut = int(0.9 * len(rows))
    conv = lambda rs: [{"text": normalize(r["text"], romanize=roman, fix_spelling=roman), "aspects": r["aspects"]} for r in rs]
    tr, va, te = conv(rows[:cut]), conv(rows[cut:]), conv(test)
    if n:
        tr, va, te = tr[:n], va[:n], te[:n]
    return tr, va, te


class TokenTask:
    def __init__(self, tok, data, max_len):
        self.tok, self.max_len = tok, max_len
        self.labels = sorted({t for seq in data[1] for t in seq})
        self.l2i = {l: i for i, l in enumerate(self.labels)}

    def encode(self, sents, tags):
        items = []
        for words, labs in zip(sents, tags):
            enc = self.tok(words, is_split_into_words=True, truncation=True, max_length=self.max_len)
            wids = enc.word_ids()
            y, prev = [], None
            for w in wids:
                y.append(-100 if w is None or w == prev else self.l2i.get(labs[w], 0))
                prev = w
            items.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": y, "word_ids": wids, "n_words": len(words)})
        return items

    def collate(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        pad = self.tok.pad_token_id or 0
        ids = torch.tensor([b["input_ids"] + [pad] * (m - len(b["input_ids"])) for b in batch])
        att = torch.tensor([b["attention_mask"] + [0] * (m - len(b["attention_mask"])) for b in batch])
        lab = torch.tensor([b["labels"] + [-100] * (m - len(b["labels"])) for b in batch])
        return {"input_ids": ids, "attention_mask": att, "labels": lab}

    def decode(self, logits, items):
        out = []
        for lg, it in zip(logits, items):
            pred = ["O"] * it["n_words"] if "O" in self.labels else [self.labels[0]] * it["n_words"]
            prev = None
            for pos, w in enumerate(it["word_ids"]):
                if w is not None and w != prev and pos < len(lg):
                    pred[w] = self.labels[int(np.argmax(lg[pos]))]
                prev = w
            out.append(pred)
        return out


class PairTask:
    """acsc: (text, aspect) -> polarity. acd: text -> multi-hot aspects."""

    def __init__(self, tok, task, train_rows, max_len):
        self.tok, self.task, self.max_len = tok, task, max_len
        if task == "acsc":
            self.labels = sorted({p for r in train_rows for _, p in r["aspects"]})
        else:
            self.labels = sorted({a for r in train_rows for a, _ in r["aspects"]})
        self.l2i = {l: i for i, l in enumerate(self.labels)}

    def examples(self, rows):
        if self.task == "acsc":
            return [(r["text"], a, p) for r in rows for a, p in r["aspects"]]
        return [(r["text"], None, [a for a, _ in r["aspects"]]) for r in rows]

    def encode(self, rows):
        items = []
        for text, aspect, y in self.examples(rows):
            if aspect is None:
                enc = self.tok(text, truncation=True, max_length=self.max_len)
                lab = [0.0] * len(self.labels)
                for a in y:
                    if a in self.l2i:
                        lab[self.l2i[a]] = 1.0
            else:
                enc = self.tok(text, aspect.replace("/", " "), truncation=True, max_length=self.max_len)
                lab = self.l2i.get(y, 0)
            items.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": lab, "gold": y})
        return items

    def collate(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        pad = self.tok.pad_token_id or 0
        ids = torch.tensor([b["input_ids"] + [pad] * (m - len(b["input_ids"])) for b in batch])
        att = torch.tensor([b["attention_mask"] + [0] * (m - len(b["attention_mask"])) for b in batch])
        dtype = torch.float if self.task == "acd" else torch.long
        return {"input_ids": ids, "attention_mask": att, "labels": torch.tensor([b["labels"] for b in batch], dtype=dtype)}

    def decode(self, logits):
        if self.task == "acd":
            out = []
            for lg in logits:
                picked = [self.labels[i] for i, v in enumerate(lg) if v > 0]
                out.append(picked or [self.labels[int(np.argmax(lg))]])
            return out
        return [self.labels[int(np.argmax(lg))] for lg in logits]


def run_epoch_eval(model, loader, device):
    model.eval()
    outs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outs.extend(model(**batch).logits.float().cpu().numpy())
    return outs


def score(task, spec, logits, items, gold):
    if task in ("lid", "ner"):
        pred = spec.decode(logits, items)
        if task == "lid":
            s = token_scores(gold, pred)
            return {"accuracy": s["accuracy"], "macro_f1": s["macro_f1"]}, s["macro_f1"]
        s = span_scores(gold, pred)
        return {"precision": s["micro"]["precision"], "recall": s["micro"]["recall"], "f1": s["micro"]["f1"]}, s["micro"]["f1"]
    pred = spec.decode(logits)
    if task == "acd":
        s = multilabel_scores(gold, pred)
        return {"precision": s["precision"], "recall": s["recall"], "f1": s["f1"]}, s["f1"]
    s = classification_scores(gold, pred)
    return {"accuracy": s["accuracy"], "macro_f1": s["macro_f1"]}, s["macro_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["lid", "ner", "acd", "acsc"])
    ap.add_argument("--model", default="l3cube-pune/hing-mbert-mixed")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--no-romanize", action="store_true", help="ACSA-Mix: keep Devanagari (use with mixed-script models)")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap examples per split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--save-model", action="store_true")
    args = ap.parse_args()
    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} task={args.task} model={args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    if args.task in ("lid", "ner"):
        tr, va, te = token_data(args.task, args.limit)
        spec = TokenTask(tok, tr, args.max_len)
        tr_items, va_items, te_items = spec.encode(*tr), spec.encode(*va), spec.encode(*te)
        va_gold, te_gold = va[1], te[1]
        model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=len(spec.labels))
    else:
        tr, va, te = acsa_data(not args.no_romanize, args.limit)
        spec = PairTask(tok, args.task, tr, args.max_len)
        tr_items, va_items, te_items = spec.encode(tr), spec.encode(va), spec.encode(te)
        va_gold, te_gold = [i["gold"] for i in va_items], [i["gold"] for i in te_items]
        kw = {"problem_type": "multi_label_classification"} if args.task == "acd" else {}
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(spec.labels), **kw)

    model.to(device)
    train_loader = DataLoader(tr_items, batch_size=args.batch_size, shuffle=True, collate_fn=spec.collate)
    eval_loader = lambda items: DataLoader(items, batch_size=args.batch_size * 2, collate_fn=spec.collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device == "cuda")

    best, best_state, history, t0 = -1.0, None, [], time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", enabled=args.fp16 and device == "cuda"):
                loss = model(**batch).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        va_scores, key = score(args.task, spec, run_epoch_eval(model, eval_loader(va_items), device), va_items, va_gold)
        history.append({"epoch": ep, "train_loss": total / max(len(train_loader), 1), "val": va_scores})
        print(f"epoch {ep}: loss={total / max(len(train_loader), 1):.4f} val={va_scores}", flush=True)
        if key > best:
            best = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    te_scores, _ = score(args.task, spec, run_epoch_eval(model, eval_loader(te_items), device), te_items, te_gold)
    print(f"TEST {args.task} {args.model}: {te_scores}", flush=True)
    out = ROOT / "results" / "transformers"
    out.mkdir(parents=True, exist_ok=True)
    name = args.model.strip("/").replace("/", "__")
    rec = {"task": args.task, "model": args.model, "romanized": not args.no_romanize, "epochs": args.epochs, "lr": args.lr,
           "train_size": len(tr_items), "test": te_scores, "history": history, "minutes": round((time.time() - t0) / 60, 1), "device": device}
    (out / f"{args.task}__{name}.json").write_text(json.dumps(rec, indent=2))
    if args.save_model:
        dest = ROOT / "models" / "transformers" / f"{args.task}__{name}"
        model.save_pretrained(dest)
        tok.save_pretrained(dest)
        (dest / "labels.json").write_text(json.dumps(spec.labels))


if __name__ == "__main__":
    main()
