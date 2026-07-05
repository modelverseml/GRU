"""
Streamlit front-end for the GRU sentiment app (Streamlit Cloud entry point).

It reuses the exact same prediction core as the FastAPI backend
(code/backend/predictor.py) -- one review in, every model's prediction out.

For the cloud deploy we run the lightweight set that actually fits the free tier
and installs cleanly: PyTorch + the from-scratch NumPy RNN, across the three
static encoders (word2vec / fastText / GloVe) = 6 models. TensorFlow is dropped
on deploy (its wheels are huge and unavailable on Streamlit's Python), and BERT
is skipped (a ~400MB transformer downloaded at runtime). Both still run locally.

Tune via env vars / Streamlit secrets (Settings -> Secrets):
  - add bert:        VRNN_ENCODERS=word2vec,fasttext,glove,bert  (needs transformers)
  - add tensorflow:  VRNN_FRAMEWORKS=pytorch,tensorflow,manual   (needs tensorflow)
"""

import os
import sys
from pathlib import Path

# choose which models run BEFORE importing predictor (it reads these at import).
# default = the lean set that deploys reliably; override via Streamlit Secrets.
os.environ.setdefault("VRNN_ENCODERS", "word2vec,fasttext,glove")
os.environ.setdefault("VRNN_FRAMEWORKS", "pytorch,manual")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code" / "backend"))

import streamlit as st
import predictor

# ---- display config ----
ENCODER_ORDER = ["word2vec", "fasttext", "glove", "bert"]
LABEL_ORDER = ["negative", "neutral", "positive"]
SENTIMENT = {
    "positive": {"color": "green", "emoji": "😊"},
    "neutral": {"color": "orange", "emoji": "😐"},
    "negative": {"color": "red", "emoji": "😞"},
}
ENCODER_META = {
    "word2vec": "Google News · 300d",
    "fasttext": "Wiki-news · 300d",
    "glove": "Wikipedia · 300d",
    "bert": "Transformer · 768d",
}
FRAMEWORK_ICON = {"pytorch": "🔥", "tensorflow": "🧠", "manual": "🧮"}
SAMPLES = [
    "this product is absolutely amazing, i love it",
    "complete waste of money, broke after one day",
    "it works fine, nothing special though",
]


@st.cache_data(show_spinner=False)
def run_predict(text):
    # cached so repeating the same review is instant; first call loads the models
    return predictor.predict(text)


def consensus(results):
    tally = {}
    for r in results:
        tally[r["label"]] = tally.get(r["label"], 0) + 1
    label, count = max(tally.items(), key=lambda kv: kv[1])
    return label, count, len(results)


st.set_page_config(page_title="GRU Sentiment", page_icon="🧠", layout="wide")

st.title("GRU Sentiment")
st.caption(
    "One review, multiple embeddings × implementations of the same GRU "
    "(from scratch in NumPy, plus PyTorch). See how they compare."
)

# keep the chosen sample text across reruns
if "text" not in st.session_state:
    st.session_state.text = SAMPLES[0]

cols = st.columns(len(SAMPLES))
for i, (c, s) in enumerate(zip(cols, SAMPLES)):
    if c.button(f"Sample {i + 1}", use_container_width=True):
        st.session_state.text = s

text = st.text_area("Review text", key="text", height=90)
go = st.button("Analyze sentiment", type="primary")

if go and text.strip():
    with st.spinner("Running the models…"):
        results = run_predict(text.strip())

    if not results:
        st.error("No models available. Did you build the artifacts (run_pipeline.py)?")
        st.stop()

    label, count, total = consensus(results)
    s = SENTIMENT[label]
    st.markdown(f"### Consensus: :{s['color']}[{s['emoji']} {label}]")
    st.caption(f"{count} of {total} models agree")
    st.divider()

    present = [e for e in ENCODER_ORDER if any(r["encoder"] == e for r in results)]
    for enc in present:
        rows = [r for r in results if r["encoder"] == enc]
        st.markdown(f"#### {enc}  ·  <span style='color:gray'>{ENCODER_META.get(enc, '')}</span>",
                    unsafe_allow_html=True)
        mcols = st.columns(len(rows))
        for c, r in zip(mcols, rows):
            sr = SENTIMENT[r["label"]]
            with c:
                st.markdown(f"**{FRAMEWORK_ICON.get(r['framework'], '')} {r['framework']}**")
                st.markdown(f":{sr['color']}[{sr['emoji']} **{r['label']}**]")
                st.metric("confidence", f"{r['confidence'] * 100:.1f}%")
                for lbl in LABEL_ORDER:
                    p = r["probabilities"].get(lbl, 0.0)
                    st.progress(min(max(p, 0.0), 1.0), text=f"{lbl} · {p * 100:.1f}%")
        st.divider()
elif go:
    st.warning("Type a review first.")
