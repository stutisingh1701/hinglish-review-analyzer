import re

import sklearn_crfsuite

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def shape(w):
    s = re.sub(r"[A-Z]", "X", w)
    s = re.sub(r"[a-z]", "x", s)
    s = re.sub(r"\d", "d", s)
    s = re.sub(r"[ऀ-ॿ]", "h", s)
    return re.sub(r"(.)\1+", r"\1\1", s)


def token_features(w):
    lw = w.lower()
    f = {
        "bias": 1.0,
        "w": lw,
        "shape": shape(w),
        "len": min(len(w), 12),
        "is_title": w.istitle(),
        "is_upper": w.isupper() and len(w) > 1,
        "is_digit": w.isdigit(),
        "has_digit": any(c.isdigit() for c in w),
        "is_hash": w.startswith("#"),
        "is_user": w.startswith("@"),
        "is_punct": bool(re.fullmatch(r"[^\w\s]+", w)),
        "deva": bool(DEVANAGARI.search(w)),
    }
    for n in range(1, 5):
        if len(lw) >= n:
            f[f"pre{n}"] = lw[:n]
            f[f"suf{n}"] = lw[-n:]
    return f


def sent2features(sent, extra=None, window=2):
    base = [token_features(w) for w in sent]
    feats = []
    for i in range(len(sent)):
        f = dict(base[i])
        if extra is not None:
            for k, v in extra[i].items():
                f[k] = v
        for d in range(1, window + 1):
            for sign, pos in (("-", i - d), ("+", i + d)):
                if 0 <= pos < len(sent):
                    nb = base[pos]
                    f[f"{sign}{d}:w"] = nb["w"]
                    f[f"{sign}{d}:suf3"] = nb.get("suf3", nb["w"])
                    f[f"{sign}{d}:shape"] = nb["shape"]
                    if extra is not None:
                        for k, v in extra[pos].items():
                            f[f"{sign}{d}:{k}"] = v
                else:
                    f["BOS" if sign == "-" else "EOS"] = True
        feats.append(f)
    return feats


class CRFTagger:
    def __init__(self, c1=0.1, c2=0.05, max_iterations=100, window=2):
        self.window = window
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
        )

    def _x(self, sentences, extras):
        if extras is None:
            extras = [None] * len(sentences)
        return [sent2features(s, e, self.window) for s, e in zip(sentences, extras)]

    def fit(self, sentences, tags, extras=None):
        self.model.fit(self._x(sentences, extras), tags)
        return self

    def predict(self, sentences, extras=None):
        return self.model.predict(self._x(sentences, extras))

    def predict_marginals(self, sentences, extras=None):
        return self.model.predict_marginals(self._x(sentences, extras))
