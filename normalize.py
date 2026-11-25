import re
import unicodedata
from functools import lru_cache
from pathlib import Path

LEXICON_DIR = Path(__file__).resolve().parent.parent / "data" / "lexicons"

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
URL = re.compile(r"https?://\S+|www\.\S+")
USER = re.compile(r"@\w+")
ELONGATION = re.compile(r"(\w)\1{2,}")
TOKEN = re.compile(
    r"<url>|<user>|emo_pos|emo_neg"
    r"|[ऀ-ॿ]+"
    r"|\d+(?:[.,]\d+)*(?:mah|mp|gb|tb|hz|k|rs|w)?"
    r"|[a-zA-Z]+(?:'[a-zA-Z]+)?"
    r"|[^\s\w]",
    re.IGNORECASE,
)

POS_EMOJI = "😍🥰😘❤💕💖👍👌🔥💯😊😁😀🙂🤩✨⭐👏💪"
NEG_EMOJI = "😡😠🤬👎😞😢😭💔😤😒🙄🤮😫😩😑"

CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
}
NUKTA_FORMS = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "फ": "f", "ड": "d", "ढ": "dh"}
PRECOMPOSED_NUKTA = {
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z",
    "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y",
}
MATRAS = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॅ": "e", "ॉ": "o",
}
VOWELS = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e",
}
DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}
VIRAMA, NUKTA, ANUSVARA, CHANDRABINDU, VISARGA = "्", "़", "ं", "ँ", "ः"
INHERENT = "<a>"


def _load_tsv(name):
    path = LEXICON_DIR / name
    table = {}
    if not path.exists():
        return table
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            table[parts[0].strip()] = parts[1].strip()
    return table


@lru_cache(maxsize=1)
def translit_exceptions():
    return _load_tsv("devanagari_exceptions.tsv")


@lru_cache(maxsize=1)
def spelling_variants():
    table = {}
    for canonical, variants in _load_tsv("spelling_variants.tsv").items():
        for v in variants.split(","):
            v = v.strip()
            if v:
                table[v] = canonical
    return table


def _segments(word):
    segs = []
    for ch in word:
        if ch in CONSONANTS:
            segs.append({"c": CONSONANTS[ch], "raw": ch, "v": INHERENT, "coda": "", "halant": False})
        elif ch in PRECOMPOSED_NUKTA:
            segs.append({"c": PRECOMPOSED_NUKTA[ch], "raw": ch, "v": INHERENT, "coda": "", "halant": False})
        elif ch == NUKTA and segs and segs[-1]["raw"] in NUKTA_FORMS:
            segs[-1]["c"] = NUKTA_FORMS[segs[-1]["raw"]]
        elif ch in MATRAS and segs:
            segs[-1]["v"] = MATRAS[ch]
        elif ch == VIRAMA and segs:
            segs[-1]["v"] = ""
            segs[-1]["halant"] = True
        elif ch in VOWELS:
            segs.append({"c": "", "raw": ch, "v": VOWELS[ch], "coda": "", "halant": False})
        elif ch in (ANUSVARA, CHANDRABINDU) and segs:
            segs[-1]["coda"] += "n"
        elif ch == VISARGA and segs:
            segs[-1]["coda"] += "h"
        elif ch in DIGITS:
            segs.append({"c": DIGITS[ch], "raw": ch, "v": "", "coda": "", "halant": False})
    return segs


def _has_vowel(seg):
    return seg["v"] not in ("", None)


def _delete_schwas(segs):
    n = len(segs)
    if n == 0:
        return segs
    last = segs[-1]
    if n > 1 and last["v"] == INHERENT and not last["coda"] and not segs[-2]["halant"]:
        last["v"] = ""
    for i in range(n - 2, 0, -1):
        s = segs[i]
        if s["v"] != INHERENT or s["coda"] or not s["c"]:
            continue
        prev, nxt = segs[i - 1], segs[i + 1]
        if _has_vowel(prev) and nxt["c"] and _has_vowel(nxt):
            s["v"] = ""
    return segs


def transliterate_word(word):
    exc = translit_exceptions()
    if word in exc:
        return exc[word]
    segs = _delete_schwas(_segments(word))
    out = "".join(s["c"] + (s["v"].replace(INHERENT, "a") if s["v"] else "") + s["coda"] for s in segs)
    out = out.replace("chchh", "cch").replace("chchh", "cch")
    return out


def transliterate(text):
    return re.sub(r"[ऀ-ॿ]+", lambda m: transliterate_word(m.group(0)), text).replace("।", ".")


def has_devanagari(text):
    return bool(DEVANAGARI.search(text))


def _emoji_tokens(text):
    out = []
    for ch in text:
        if ch in POS_EMOJI:
            out.append(" emo_pos ")
        elif ch in NEG_EMOJI:
            out.append(" emo_neg ")
        else:
            out.append(ch)
    return "".join(out)


def normalize(text, romanize=True, fix_spelling=True):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = URL.sub(" <url> ", text)
    text = USER.sub(" <user> ", text)
    text = text.replace("#", " ")
    text = _emoji_tokens(text)
    if romanize:
        text = transliterate(text)
    text = text.lower()
    text = ELONGATION.sub(r"\1\1", text)
    if fix_spelling:
        variants = spelling_variants()
        text = " ".join(_canonical(t, variants) for t in tokenize(text))
    return re.sub(r"\s+", " ", text).strip()


def _canonical(tok, variants):
    if tok in variants:
        return variants[tok]
    single = re.sub(r"([a-z])\1+", r"\1", tok)
    if single != tok and single in variants:
        return variants[single]
    return tok


def tokenize(text):
    return TOKEN.findall(text)


def split_clauses(text):
    parts = re.split(
        r"[.!?;,]+|\b(?:but|lekin|par|magar|however|although|though|whereas|jabki|kintu|parantu|and also)\b",
        text,
    )
    return [p.strip() for p in parts if p and p.strip()]
