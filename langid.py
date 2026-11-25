import math
import re
from collections import Counter, defaultdict

from .crf_tagger import CRFTagger

ALPHA = re.compile(r"[a-zA-Zऀ-ॿ]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


class CharNgramLM:
    """Character n-gram language model, interpolated over orders 1..n,
    each order add-k smoothed (Lidstone)."""

    def __init__(self, n=4, k=0.5, lambdas=None):
        self.n = n
        self.k = k
        self.lambdas = lambdas or [0.1, 0.2, 0.3, 0.4][:n]
        s = sum(self.lambdas)
        self.lambdas = [l / s for l in self.lambdas]

    def _pad(self, word):
        return "^" * (self.n - 1) + word.lower() + "$"

    def fit(self, words):
        self.counts = [defaultdict(Counter) for _ in range(self.n)]
        chars = set()
        for w in words:
            p = self._pad(w)
            chars.update(p)
            for i in range(self.n - 1, len(p)):
                for order in range(1, self.n + 1):
                    ctx = p[i - order + 1:i]
                    self.counts[order - 1][ctx][p[i]] += 1
        self.V = len(chars) + 1
        self.totals = [{ctx: sum(c.values()) for ctx, c in level.items()} for level in self.counts]
        return self

    def logprob(self, word):
        p = self._pad(word)
        lp = 0.0
        for i in range(self.n - 1, len(p)):
            prob = 0.0
            for order in range(1, self.n + 1):
                ctx = p[i - order + 1:i]
                c = self.counts[order - 1].get(ctx)
                num = (c[p[i]] if c else 0) + self.k
                den = self.totals[order - 1].get(ctx, 0) + self.k * self.V
                prob += self.lambdas[order - 1] * num / den
            lp += math.log(prob)
        return lp


class NgramLID:
    """Word-level language ID: one char n-gram LM per language + class prior.
    Context-free baseline (Unit III: n-gram LMs with smoothing)."""

    def __init__(self, n=4, k=0.5):
        self.n, self.k = n, k

    def fit(self, sentences, tags):
        by_lang = defaultdict(list)
        for s, t in zip(sentences, tags):
            for w, l in zip(s, t):
                by_lang[l].append(w)
        total = sum(len(v) for v in by_lang.values())
        self.langs = sorted(by_lang)
        self.prior = {l: math.log(len(by_lang[l]) / total) for l in self.langs}
        self.lms = {l: CharNgramLM(self.n, self.k).fit(by_lang[l]) for l in self.langs}
        return self

    def scores(self, word):
        return {l: self.prior[l] + self.lms[l].logprob(word) for l in self.langs}

    def predict_word(self, word):
        if DEVANAGARI.search(word) and "HI" in self.langs:
            return "HI"
        s = self.scores(word)
        return max(s, key=s.get)

    def predict(self, sentences):
        return [[self.predict_word(w) for w in s] for s in sentences]

    def features(self, sent):
        out = []
        for w in sent:
            s = self.scores(w)
            best = max(s, key=s.get)
            ordered = sorted(s.values(), reverse=True)
            margin = ordered[0] - ordered[1] if len(ordered) > 1 else 0.0
            f = {"lm_best": best, "lm_margin": min(round(margin), 10)}
            per_char = {l: v / (len(w) + 1) for l, v in s.items()}
            for l, v in per_char.items():
                f[f"lm_{l}"] = v
            out.append(f)
        return out


class CRFLID:
    """Context-aware LID: CRF over word/char features plus the n-gram LM scores."""

    def __init__(self, ngram=None, max_iterations=60):
        self.ngram = ngram
        self.crf = CRFTagger(max_iterations=max_iterations, window=2)

    def _extras(self, sentences):
        if self.ngram is None:
            return None
        return [self.ngram.features(s) for s in sentences]

    def fit(self, sentences, tags):
        self.crf.fit(sentences, tags, self._extras(sentences))
        return self

    def predict(self, sentences):
        return self.crf.predict(sentences, self._extras(sentences))


def tag_with_other(model, tokens):
    if not tokens:
        return []
    pred = model.predict([tokens])[0]
    return [p if ALPHA.search(t) and not t.startswith(("<", "emo_")) else "OTHER" for t, p in zip(tokens, pred)]


def code_mixing_index(langs):
    """CMI (Das & Gamback, 2014): 100 * (1 - max_lang_count / (n - u)),
    u = language-independent tokens."""
    n = len(langs)
    counts = Counter(l for l in langs if l not in ("OTHER", "O", "univ", "ne"))
    u = n - sum(counts.values())
    if n == u or not counts:
        return 0.0
    return 100.0 * (1 - max(counts.values()) / (n - u))
