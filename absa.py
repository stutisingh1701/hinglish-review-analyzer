import json
import math
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from .normalize import LEXICON_DIR, _load_tsv, split_clauses, tokenize

SEP = " ||| "
NEGATORS = {"nahi", "na", "not", "no", "never", "mat", "nope", "nahin", "didnt", "dont", "isnt", "wasnt", "doesnt", "n't"}
INTENSIFIERS = {"bahut": 1.5, "very": 1.5, "too": 1.3, "ekdum": 1.5, "bilkul": 1.3, "kaafi": 1.2, "really": 1.3, "so": 1.3, "super": 1.5, "itna": 1.3}


def make_classifier(name):
    if name == "nb":
        return ComplementNB(alpha=0.3)
    if name == "lr":
        return LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
    if name == "svm":
        return LinearSVC(C=0.5, class_weight="balanced")
    raise ValueError(name)


def text_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, token_pattern=r"[^\s]+")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])


class AspectCategoryDetector:
    """Multi-label aspect category detection: TF-IDF (word + char n-grams), one-vs-rest."""

    def __init__(self, clf="svm"):
        self.clf_name = clf

    def fit(self, texts, aspect_sets):
        self.mlb = MultiLabelBinarizer()
        Y = self.mlb.fit_transform(aspect_sets)
        self.model = make_pipeline(text_features(), OneVsRestClassifier(make_classifier(self.clf_name)))
        self.model.fit(texts, Y)
        return self

    def _scores(self, texts):
        est = self.model
        if hasattr(est, "decision_function"):
            try:
                return est.decision_function(texts), 0.0
            except AttributeError:
                pass
        return est.predict_proba(texts), 0.5

    def predict(self, texts):
        scores, thr = self._scores(texts)
        out = []
        for row in np.atleast_2d(scores):
            picked = [self.mlb.classes_[i] for i, s in enumerate(row) if s > thr]
            if not picked:
                picked = [self.mlb.classes_[int(np.argmax(row))]]
            out.append(picked)
        return out


def _pair_analyzer(doc):
    aspect, text = doc.split(SEP, 1)
    toks = text.split()
    feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    feats += [f"{aspect}::{t}" for t in toks]
    feats.append(f"ASPECT={aspect}")
    neg = False
    for t in toks:
        if t in NEGATORS:
            neg = True
            feats.append("HAS_NEG")
        elif neg:
            feats.append(f"NEG_{t}")
    return feats


def _strip_aspect(doc):
    return doc.split(SEP, 1)[1]


class AspectSentimentClassifier:
    """Sentiment for a (text, aspect) pair. Aspect-crossed features let the
    linear model learn aspect-specific opinion words."""

    def __init__(self, clf="svm"):
        self.clf_name = clf

    @staticmethod
    def pair(text, aspect):
        return f"{aspect}{SEP}{text}"

    def fit(self, texts, aspects, labels):
        docs = [self.pair(t, a) for t, a in zip(texts, aspects)]
        feats = FeatureUnion([
            ("pair", TfidfVectorizer(analyzer=_pair_analyzer, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, preprocessor=_strip_aspect)),
        ])
        self.model = make_pipeline(feats, make_classifier(self.clf_name))
        self.model.fit(docs, labels)
        self.classes_ = list(self.model.classes_)
        return self

    def predict(self, texts, aspects):
        return list(self.model.predict([self.pair(t, a) for t, a in zip(texts, aspects)]))

    def predict_proba(self, texts, aspects):
        docs = [self.pair(t, a) for t, a in zip(texts, aspects)]
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(docs)
        d = np.atleast_2d(self.model.decision_function(docs))
        e = np.exp(d - d.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


class LexiconSentiment:
    def __init__(self):
        self.lex = {k: float(v) for k, v in _load_tsv("sentiment_lexicon.tsv").items()}

    def score(self, text):
        toks = tokenize(text) if not isinstance(text, list) else text
        total = 0.0
        hits = []
        for i, t in enumerate(toks):
            if t not in self.lex:
                continue
            s = self.lex[t]
            window = toks[max(0, i - 3):i] + toks[i + 1:i + 3]
            if any(w in NEGATORS for w in window):
                s = -s * 0.8
            if i > 0 and toks[i - 1] in INTENSIFIERS:
                s *= INTENSIFIERS[toks[i - 1]]
            total += s
            hits.append((t, round(s, 2)))
        return total, hits


GENERIC_TERMS = {"fast", "slow", "smooth", "worth", "value", "cheap", "heavy", "premium", "look", "looks", "touch", "game", "body"}


class ProductAspectExtractor:
    """Lexicon-based aspect term extraction for the product-review domain.
    Longest match first ('fast charging' -> battery); generic words like 'fast'
    only count when nothing more specific is in the clause."""

    def __init__(self, path=None):
        path = path or LEXICON_DIR / "product_aspects.json"
        self.lex = json.loads(open(path, encoding="utf-8").read())
        self.terms = sorted(
            ((tuple(term.split()), aspect) for aspect, terms in self.lex.items() for term in terms),
            key=lambda x: -len(x[0]),
        )

    def find(self, tokens):
        used = [False] * len(tokens)
        found = []
        for term, aspect in self.terms:
            n = len(term)
            for i in range(len(tokens) - n + 1):
                if tuple(tokens[i:i + n]) == term and not any(used[i:i + n]):
                    for j in range(i, i + n):
                        used[j] = True
                    found.append({"aspect": aspect, "term": " ".join(term), "start": i, "end": i + n})
        specific = [f for f in found if f["term"] not in GENERIC_TERMS]
        if specific:
            found = specific
        return sorted(found, key=lambda f: f["start"])


class ClauseSentiment:
    """Clause-level polarity for open-domain aspects: combines a statistical
    model trained on ACSA-Mix with the Hinglish lexicon (negation-aware)."""

    def __init__(self, model=None, aspect_hint="anecdotes/miscellaneous", lex_weight=0.6):
        self.model = model
        self.aspect_hint = aspect_hint
        self.lex = LexiconSentiment()
        self.lex_weight = lex_weight

    def predict(self, clause):
        lex_score, hits = self.lex.score(clause)
        model_score, probs = 0.0, {}
        if self.model is not None:
            p = self.model.predict_proba([clause], [self.aspect_hint])[0]
            probs = dict(zip(self.model.classes_, p))
            model_score = probs.get("positive", 0) - probs.get("negative", 0)
        combined = (1 - self.lex_weight) * model_score + self.lex_weight * math.tanh(lex_score)
        if combined > 0.15:
            label = "positive"
        elif combined < -0.15:
            label = "negative"
        else:
            label = "neutral"
        return {"label": label, "score": round(float(combined), 3), "lexicon_hits": hits, "model_probs": {k: round(float(v), 3) for k, v in probs.items()}}


def product_absa(text_norm, extractor, clause_model):
    results = []
    for clause in split_clauses(text_norm):
        toks = tokenize(clause)
        mentions = extractor.find(toks)
        if not mentions:
            continue
        sent = clause_model.predict(clause)
        for m in mentions:
            shown = " ".join(t for t in clause.split() if t not in ("emo_pos", "emo_neg"))
            results.append({"aspect": m["aspect"], "term": m["term"], "clause": shown, **sent})
    seen, out = set(), []
    for r in results:
        key = (r["aspect"], r["clause"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
