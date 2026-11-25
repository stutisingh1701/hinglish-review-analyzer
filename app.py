"""Streamlit demo:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

from hinglish.pipeline import HinglishAnalyzer

ROOT = Path(__file__).resolve().parent
LANG_COLOR = {"HI": "#f4a261", "EN": "#2a9d8f", "OTHER": "#adb5bd"}
POL_COLOR = {"positive": "#2a9d8f", "negative": "#e76f51", "neutral": "#8d99ae"}

st.set_page_config(page_title="Hinglish Review Analyzer", page_icon="🗣️", layout="wide")


@st.cache_resource
def get_analyzer():
    return HinglishAnalyzer()


def chip(text, color):
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:10px;margin:2px;display:inline-block">{text}</span>'


an = get_analyzer()
if an.lid is None:
    st.error("No trained models found. Run `python scripts/train_all.py` first.")
    st.stop()

st.title("Hinglish Review Analyzer")
st.caption("Code-mixed Hindi-English: normalization → language ID → NER → aspect sentiment → opinion summary")
tab1, tab2, tab3 = st.tabs(["Single review", "Summarize many reviews", "Model results"])

with tab1:
    examples = [
        "Camera bohot mast hai but battery jaldi khatam ho jati h 😞",
        "Display ekdum zabardast, gaming me thoda heat hota hai",
        "कैमरा अच्छा है लेकिन बैटरी बेकार है",
        "Samsung ka phone Mumbai me liya, delivery fast thi par packaging ghatiya thi",
    ]
    pick = st.selectbox("Try an example", examples)
    text = st.text_area("Review", pick, height=90)
    domain = st.radio("Aspect set", ["product", "restaurant"], horizontal=True,
                      help="product = phone aspects (lexicon + clause sentiment); restaurant = ACSA-Mix categories (trained classifiers)")
    if text.strip():
        out = an.analyze(text, domain=domain)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("1. Normalized")
            st.code(out["normalized"], language=None)
            st.subheader("2. Word-level language")
            st.markdown(" ".join(chip(t, LANG_COLOR.get(l, "#999")) for t, l in zip(out["tokens"], out["lang"])), unsafe_allow_html=True)
            st.caption(f"Code-Mixing Index: {out['cmi']}  ·  orange = Hindi, teal = English, grey = other")
            if out["pos"]:
                st.subheader("POS tags")
                st.write(" ".join(f"{t}/{p}" for t, p in zip(out["tokens"], out["pos"])))
            st.subheader("3. Named entities")
            if out["entities"]:
                st.markdown(" ".join(chip(f'{e["text"]} · {e["type"]}', "#6c757d") for e in out["entities"]), unsafe_allow_html=True)
            else:
                st.write("None found")
        with c2:
            st.subheader("4. Aspect sentiment")
            if out["aspects"]:
                for a in out["aspects"]:
                    st.markdown(chip(f'{a["aspect"]}: {a["label"]}', POL_COLOR.get(a["label"], "#999")) + f' <span style="color:#666">“{a["clause"]}”</span>', unsafe_allow_html=True)
                with st.expander("Details"):
                    st.json(out["aspects"])
            else:
                st.write("No aspect found")

with tab2:
    st.write("Upload a CSV with a `text` column, or paste one review per line.")
    up = st.file_uploader("CSV", type="csv")
    pasted = st.text_area("Reviews", "\n".join([
        "Camera bohot mast hai but battery jaldi khatam ho jati h",
        "Battery backup bekaar hai yaar, 4 ghante bhi nahi chalti",
        "Display ekdum zabardast hai",
        "Price ke hisaab se paisa vasool phone hai",
        "Camera quality badhiya hai, night photos bhi acche",
        "Phone gaming me bahut garam hota hai",
    ]), height=170)
    reviews = pd.read_csv(up)["text"].dropna().astype(str).tolist() if up else [l for l in pasted.splitlines() if l.strip()]
    if st.button("Summarize", type="primary") and reviews:
        analyses, summ, text_summary = an.summarize(reviews)
        if not summ:
            st.warning("No aspects found in these reviews.")
        else:
            df = pd.DataFrame(summ)
            st.subheader("Opinion summary")
            st.text(text_summary)
            st.bar_chart(df.set_index("aspect")[["positive", "negative", "neutral"]], color=["#2a9d8f", "#e76f51", "#8d99ae"], horizontal=True)
            st.dataframe(df[["aspect", "mentions", "positive", "negative", "neutral", "verdict"]], hide_index=True)
            rows = [{"review": a["input"], "aspect": x["aspect"], "sentiment": x["label"]} for a in analyses for x in a["aspects"]]
            st.download_button("Download per-review results", pd.DataFrame(rows).to_csv(index=False), "aspect_results.csv")

with tab3:
    md = ROOT / "results" / "results.md"
    st.markdown(md.read_text(encoding="utf-8") if md.exists() else "Run `python scripts/train_all.py` to generate results.")
