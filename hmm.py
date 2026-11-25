import math
import re
from collections import Counter, defaultdict

import numpy as np

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def word_class(word):
    if re.fullmatch(r"\d+([.,:/]\d+)*", word):
        return "<NUM>"
    if word.startswith("#"):
        return "<HASH>"
    if word.startswith("@"):
        return "<USER>"
    if word.startswith("http"):
        return "<URL>"
    if re.fullmatch(r"[^\w\s]+", word):
        return "<PUNCT>"
    return None


class HMMTagger:
    """First-order (bigram) HMM tagger with add-k smoothing, suffix-based
    unknown-word emissions and Viterbi decoding."""

    def __init__(self, k=0.1, suffix_len=3, lowercase=True):
        self.k = k
        self.suffix_len = suffix_len
        self.lowercase = lowercase

    def _norm(self, w):
        cls = word_class(w)
        if cls:
            return cls
        return w.lower() if self.lowercase else w

    def _suffix_keys(self, w):
        keys = [f"suf{n}:{w[-n:]}" for n in range(1, self.suffix_len + 1) if len(w) >= n]
        keys.append("cap" if w[:1].isupper() else "nocap")
        keys.append("deva" if DEVANAGARI.search(w) else "roman")
        return keys

    def fit(self, sentences, tags):
        self.tags = sorted({t for seq in tags for t in seq})
        self.tag_index = {t: i for i, t in enumerate(self.tags)}
        T = len(self.tags)
        start = np.zeros(T)
        trans = np.zeros((T, T))
        emit = defaultdict(Counter)
        tag_count = Counter()
        word_count = Counter()
        for sent, seq in zip(sentences, tags):
            prev = None
            for w, t in zip(sent, seq):
                i = self.tag_index[t]
                if prev is None:
                    start[i] += 1
                else:
                    trans[prev, i] += 1
                prev = i
                nw = self._norm(w)
                emit[t][nw] += 1
                tag_count[t] += 1
                word_count[nw] += 1
        k = self.k
        self.log_start = np.log((start + k) / (start.sum() + k * T))
        self.log_trans = np.log((trans + k) / (trans.sum(axis=1, keepdims=True) + k * T))
        self.vocab = set(word_count)
        V = len(self.vocab) + 1
        self.log_emit = {}
        for t in self.tags:
            denom = tag_count[t] + k * V
            self.log_emit[t] = {w: math.log((c + k) / denom) for w, c in emit[t].items()}
            self.log_emit[t]["<UNK_FLOOR>"] = math.log(k / denom)

        hapax = defaultdict(Counter)
        hapax_tag = Counter()
        for t in self.tags:
            for w, c in emit[t].items():
                if word_count[w] == 1:
                    hapax_tag[t] += 1
                    for key in self._suffix_keys(w):
                        hapax[t][key] += 1
        total_hapax = sum(hapax_tag.values()) + k * T
        self.log_unk_tag = {t: math.log((hapax_tag[t] + k) / total_hapax) for t in self.tags}
        self.hapax = hapax
        self.hapax_tag = hapax_tag
        return self

    def _unk_logprob(self, w, t):
        lp = self.log_unk_tag[t]
        n = self.hapax_tag[t] + 1.0
        for key in self._suffix_keys(w):
            lp += math.log((self.hapax[t][key] + 0.5) / (n + 1.0))
        return lp

    def _emission_matrix(self, sent):
        E = np.empty((len(sent), len(self.tags)))
        for j, w in enumerate(sent):
            nw = self._norm(w)
            known = nw in self.vocab
            for i, t in enumerate(self.tags):
                table = self.log_emit[t]
                if known:
                    E[j, i] = table.get(nw, table["<UNK_FLOOR>"])
                else:
                    E[j, i] = self._unk_logprob(w, t)
        return E

    def viterbi(self, sent):
        if not sent:
            return []
        E = self._emission_matrix(sent)
        n, T = E.shape
        score = self.log_start + E[0]
        back = np.zeros((n, T), dtype=int)
        for j in range(1, n):
            cand = score[:, None] + self.log_trans
            back[j] = cand.argmax(axis=0)
            score = cand.max(axis=0) + E[j]
        best = [int(score.argmax())]
        for j in range(n - 1, 0, -1):
            best.append(int(back[j, best[-1]]))
        return [self.tags[i] for i in reversed(best)]

    def predict(self, sentences):
        return [self.viterbi(s) for s in sentences]
