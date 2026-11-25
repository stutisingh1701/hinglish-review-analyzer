from pathlib import Path

import joblib

from .absa import ClauseSentiment, ProductAspectExtractor, product_absa
from .langid import code_mixing_index, tag_with_other
from .metrics import bio_spans
from .normalize import has_devanagari, normalize, tokenize
from .summarize import opinion_summary, summary_text

MODELS = Path(__file__).resolve().parent.parent / "models"


class HinglishAnalyzer:
    """Raw review -> normalize -> tokenize -> LID -> (POS) -> NER -> aspects + sentiment."""

    def __init__(self, models_dir=MODELS):
        d = Path(models_dir)

        def load(name):
            if not (d / name).exists():
                return None
            try:
                return joblib.load(d / name)
            except Exception as e:
                raise RuntimeError(f"Could not load {name} ({e}). Your library versions differ from the ones used "
                                   f"to train it: run `python scripts/train_all.py` to retrain.") from e

        self.lid = load("lid_crf.joblib") or load("lid_hmm.joblib")
        self.ner = load("ner_crf.joblib")
        self.ner_uses_lid = (d / "ner_crf.uses_lid").exists()
        self.lid_ngram = load("lid_ngram.joblib")
        self.pos = load("pos_crf.joblib") or load("pos_hmm.joblib")
        self.acd = load("acd_svm.joblib")
        self.acsc = load("acsc_lr.joblib")
        self.extractor = ProductAspectExtractor()
        self.clause = ClauseSentiment(self.acsc)

    def analyze(self, review, domain="product"):
        norm = normalize(review)
        tokens = tokenize(norm)
        langs = tag_with_other(self.lid, tokens) if self.lid else []
        entities = []
        raw = [t for t in tokenize(review) if t.strip()]
        if self.ner and raw:
            extra = [self.lid_ngram.features(raw)] if (self.ner_uses_lid and self.lid_ngram) else None
            tags = self.ner.predict([raw], extra)[0]
            entities = [{"text": " ".join(raw[s:e]), "type": lab} for s, e, lab in bio_spans(tags)]
        pos = self.pos.predict([tokens])[0] if self.pos and tokens else []
        result = {
            "input": review,
            "had_devanagari": has_devanagari(review),
            "normalized": norm,
            "tokens": tokens,
            "lang": langs,
            "cmi": round(code_mixing_index(langs), 1) if langs else None,
            "pos": pos,
            "entities": entities,
        }
        if domain == "product":
            result["aspects"] = product_absa(norm, self.extractor, self.clause)
        else:
            cats = self.acd.predict([norm])[0] if self.acd else []
            labels = self.acsc.predict([norm] * len(cats), cats) if (self.acsc and cats) else []
            result["aspects"] = [{"aspect": str(c), "label": str(l), "clause": norm, "score": None} for c, l in zip(cats, labels)]
        return result

    def summarize(self, reviews, domain="product", top_k=2):
        analyses = [self.analyze(r, domain) for r in reviews]
        summ = opinion_summary(analyses, top_k=top_k)
        return analyses, summ, summary_text(summ)
