"""python tests/smoke_test.py  — quick end-to-end check after training."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hinglish.hmm import HMMTagger
from hinglish.metrics import bio_spans, rouge
from hinglish.normalize import normalize, transliterate
from hinglish.pipeline import HinglishAnalyzer

assert normalize("Camera bohot achhaaa h") == "camera bahut accha hai"
assert transliterate("सबसे") == "sabse" and transliterate("करता") == "karta"
assert bio_spans(["B-Per", "I-Per", "O", "B-Loc"]) == [(0, 2, "Per"), (3, 4, "Loc")]
assert rouge("battery bahut kharab hai", "battery kharab hai")["rouge1"] > 0.8

toy = HMMTagger().fit([["a", "b"], ["a", "c"]], [["X", "Y"], ["X", "Y"]])
assert toy.viterbi(["a", "z"]) == ["X", "Y"]

an = HinglishAnalyzer()
assert an.lid is not None and an.ner is not None and an.acsc is not None, "train models first: python scripts/train_all.py"
out = an.analyze("Camera bohot mast hai but battery bekaar hai")
labels = {a["aspect"]: a["label"] for a in out["aspects"]}
assert labels == {"camera": "positive", "battery": "negative"}, labels
assert set(out["lang"]) <= {"HI", "EN", "OTHER"} and "HI" in out["lang"] and "EN" in out["lang"]
ents = an.analyze("Modi ji aaj Delhi me rally karenge")["entities"]
assert any(e["type"] == "Loc" for e in ents), ents
_, summ, text = an.summarize(["Camera mast hai", "Camera bakwas hai", "Battery zabardast"])
assert {s["aspect"] for s in summ} == {"camera", "battery"}
print("All smoke tests passed.")
