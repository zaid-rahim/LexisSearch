import streamlit as st
import numpy as np
import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────────────────
# 1. Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexisSearch · Semantic Legal Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. CSS Injections
# ─────────────────────────────────────────────────────────────────────────────
_FONT = '<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">'

_CSS_GLOBAL = """
<style>
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #F8F7F3 !important;
    color: #2D333B;
    font-family: system-ui, -apple-system, sans-serif;
}
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container { padding-top: 1rem !important; }
</style>
"""

_CSS_COMPONENTS = """
<style>
.masthead {
    border-bottom: 3px solid #1B2A47;
    padding: 1.8rem 0 1.1rem 0;
    margin-bottom: 2rem;
}
.masthead-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 3px;
    color: #C8922A;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.masthead-title {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 2.5rem;
    font-weight: bold;
    color: #1B2A47;
    line-height: 1.1;
    margin: 0;
}
.masthead-sub {
    font-size: 0.93rem;
    color: #6B7280;
    margin-top: 0.45rem;
    font-style: italic;
}
.stats-row {
    display: flex;
    gap: 1.2rem;
    margin-bottom: 1.8rem;
    flex-wrap: wrap;
}
.stat-box {
    background: #FFFFFF;
    border: 1px solid #E0DDD4;
    border-top: 3px solid #1B2A47;
    padding: 0.8rem 1.2rem;
    border-radius: 2px;
    min-width: 120px;
}
.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #1B2A47;
    line-height: 1;
}
.stat-label {
    font-size: 0.72rem;
    color: #9CA3AF;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.search-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    color: #1B2A47;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    display: block;
    font-weight: 600;
}
</style>
"""

_CSS_INPUT = """
<style>
div[data-testid="stTextInput"] { margin-bottom: 0.5rem; }
div[data-testid="stTextInput"] label { display: none !important; }
div[data-baseweb="input"] { background-color: #FFFFFF !important; border-radius: 3px !important; }
div[data-baseweb="input"] > div {
    background-color: #FFFFFF !important;
    border: 2px solid #1B2A47 !important;
    border-radius: 3px !important;
    box-shadow: 0 2px 8px rgba(27,42,71,0.06) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
div[data-baseweb="input"]:focus-within > div {
    border-color: #C8922A !important;
    box-shadow: 0 0 0 3px rgba(200,146,42,0.18), 0 2px 8px rgba(27,42,71,0.08) !important;
}
div[data-baseweb="input"] input {
    font-size: 1.05rem !important;
    padding: 0.8rem 1.1rem !important;
    color: #1B2A47 !important;
    background-color: #FFFFFF !important;
    font-family: system-ui, sans-serif !important;
}
div[data-baseweb="input"] input::placeholder { color: #A0A8B4 !important; font-style: italic; }
</style>
"""

_CSS_CARDS = """
<style>
.section-header {
    font-family: Georgia, serif;
    font-size: 1.05rem;
    color: #1B2A47;
    font-weight: bold;
    border-bottom: 1px solid #E0DDD4;
    padding-bottom: 0.5rem;
    margin-bottom: 1.1rem;
}
.result-card {
    background: #FFFFFF;
    border: 1px solid #E0DDD4;
    border-left: 4px solid #1B2A47;
    border-radius: 2px 4px 4px 2px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
    transition: border-left-color 0.2s ease, box-shadow 0.2s ease;
}
.result-card:hover { border-left-color: #C8922A; box-shadow: 0 4px 18px rgba(27,42,71,0.09); }
.result-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    letter-spacing: 2px;
    color: #9CA3AF;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}
.result-text { font-size: 0.96rem; line-height: 1.65; color: #1A1D20; margin-bottom: 0.85rem; }
.result-footer { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; }
.score-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    background-color: #1B2A47;
    color: #F8F7F3;
    padding: 3px 10px;
    border-radius: 2px;
}
.score-bar-wrap { flex: 1; height: 5px; background: #E8E4DA; border-radius: 3px; overflow: hidden; min-width: 80px; max-width: 200px; }
.score-bar-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #1B2A47, #C8922A); }
.doc-id { font-family: 'JetBrains Mono', monospace; font-size: 0.67rem; color: #9CA3AF; }
.empty-state {
    background: #FFFFFF;
    border: 1px dashed #C0BBB0;
    border-radius: 4px;
    padding: 3rem 2rem;
    text-align: center;
    color: #6B7280;
    margin-top: 1rem;
}
.empty-title { font-family: Georgia, serif; font-size: 1.2rem; color: #1B2A47; margin-bottom: 0.4rem; }
.empty-examples { margin-top: 1.1rem; display: flex; flex-wrap: wrap; justify-content: center; gap: 0.5rem; }
.example-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
    background: #F8F7F3;
    border: 1px solid #D8D4CA;
    color: #1B2A47;
    padding: 4px 12px;
    border-radius: 2px;
}
.chart-panel { background: #FFFFFF; border: 1px solid #E0DDD4; border-radius: 3px; padding: 1.2rem 1.3rem; margin-top: 1rem; }
.chart-panel-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 2px;
    color: #9CA3AF;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
</style>
"""

st.markdown(_FONT, unsafe_allow_html=True)
st.markdown(_CSS_GLOBAL, unsafe_allow_html=True)
st.markdown(_CSS_COMPONENTS, unsafe_allow_html=True)
st.markdown(_CSS_INPUT, unsafe_allow_html=True)
st.markdown(_CSS_CARDS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Masthead
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div class="masthead-eyebrow">Semantic Retrieval Engine &nbsp;·&nbsp; v2.0</div>
    <div class="masthead-title">⚖️ LexisSearch</div>
    <div class="masthead-sub">Neural search across judicial holdings, precedents, and legal commentary.</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Load Resources
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading matrix indexes...")
def load_resources():
    return np.load("Doc_Embeddings.npy")

try:
    doc_embeddings = load_resources()
except FileNotFoundError:
    st.error("**Missing file:** `Doc_Embeddings.npy` not found. Place it in the working directory and restart.")
    st.stop()


@st.cache_data
def load_documents():
    try:
        df = pd.read_csv("Pakistan_News.csv")
        return df["Details"].astype(str).tolist()
    except FileNotFoundError:
        st.error("**Missing file:** `Pakistan_News.csv` not found. Please ensure it is uploaded to your repository.")
        st.stop()
    except Exception as e:
        st.error(f"Could not load documents: {e}")
        return []

documents = load_documents()

if len(documents) != doc_embeddings.shape[0]:
    st.warning(
        f"⚠️ Row count mismatch — {len(documents)} documents in CSV vs "
        f"{doc_embeddings.shape[0]} embeddings in matrix. Results may be misaligned."
    )

# ─────────────────────────────────────────────────────────────────────────────
# 5. Corpus Stats
# ─────────────────────────────────────────────────────────────────────────────
dims = doc_embeddings.shape[1] if doc_embeddings.ndim == 2 else "—"
st.markdown(f"""
<div class="stats-row">
    <div class="stat-box">
        <div class="stat-value">{len(documents):,}</div>
        <div class="stat-label">Documents</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">{dims}</div>
        <div class="stat-label">Embed Dims</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">API Server</div>
        <div class="stat-label">Encoder</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">Cosine</div>
        <div class="stat-label">Metric</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Search Input & Serverless Function
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<span class="search-label">🔍 &nbsp;Query the Corpus</span>', unsafe_allow_html=True)

query_text = st.text_input(
    label="query",
    label_visibility="collapsed",
    placeholder="e.g. right to fair trial · custodial torture · habeas corpus jurisdiction",
    key="main_query",
)

# REPLACE WITH THIS:
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-mpnet-base-v2"

# Pull the token securely from the Streamlit vault
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except KeyError:
    st.error("🔑 Database Secret 'HF_TOKEN' not found in Streamlit Secrets.")
    st.stop()
def call_embedding_api(text_query):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": text_query, "options": {"wait_for_model": True}}
    
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Hugging Face API Error {response.status_code}: {response.text}")
    return np.array(response.json())

def cosine_top_k(query_vec, doc_vecs, k=5):
    q = query_vec.reshape(1, -1)
    q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-10)
    d_norm = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    sims = np.dot(q_norm, d_norm.T).flatten()
    idx = np.argsort(sims)[::-1][:k]
    return idx, sims[idx]

# ─────────────────────────────────────────────────────────────────────────────
# 7. Results rendering
# ─────────────────────────────────────────────────────────────────────────────
if query_text.strip():
    if not documents:
        st.error("No documents loaded. Configure `load_documents()` with your corpus path.")
        st.stop()

    with st.spinner("Requesting serverless vector transformation and running search..."):
        try:
            q_emb = call_embedding_api(query_text.strip())
            top_idx, top_scores = cosine_top_k(q_emb, doc_embeddings, k=5)
        except Exception as e:
            st.error(f"Failed to reach the Inference API: {e}")
            st.stop()

    results = pd.DataFrame({
        "rank":    list(range(1, len(top_idx) + 1)),
        "index":   top_idx,
        "score":   top_scores,
        "content": [documents[i] for i in top_idx],
    })

    st.markdown("<br>", unsafe_allow_html=True)
    col_results, col_chart = st.columns([3, 2], gap="large")

    with col_results:
        st.markdown('<div class="section-header">📋 &nbsp;Top Matching Documents</div>', unsafe_allow_html=True)

        ordinals = ["1st", "2nd", "3rd", "4th", "5th"]
        for _, row in results.iterrows():
            pct = row["score"] * 100
            bar_pct = max(0, min(100, pct))
            rank_label = ordinals[int(row["rank"]) - 1]

            display_text = row["content"]
            if len(display_text) > 420:
                display_text = display_text[:420].rstrip() + "…"

            st.markdown(f"""
            <div class="result-card">
                <div class="result-rank">Result {rank_label} &nbsp;·&nbsp; Corpus Index {row['index']}</div>
                <div class="result-text">{display_text}</div>
                <div class="result-footer">
                    <span class="score-pill">{row['score']:.4f}</span>
                    <div class="score-bar-wrap">
                        <div class="score-bar-fill" style="width:{bar_pct:.1f}%"></div>
                    </div>
                    <span class="doc-id">DOC-{row['index']:05d}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_chart:
        st.markdown('<div class="section-header">📊 &nbsp;Match Distribution</div>', unsafe_allow_html=True)

        chart_df = pd.DataFrame({
            "Document": [f"DOC-{i:05d}" for i in results["index"]],
            "Similarity": results["score"].values,
        })

        st.bar_chart(
            chart_df.set_index("Document"),
            y="Similarity",
            color="#1B2A47",
            height=300,
        )

        st.markdown("""
        <div class="chart-panel">
            <div class="chart-panel-title">Score Summary</div>
        </div>
        """, unsafe_allow_html=True)

        summary_df = pd.DataFrame({
            "Metric": ["Highest", "Lowest", "Mean", "Spread"],
            "Value": [
                f"{results['score'].max():.4f}",
                f"{results['score'].min():.4f}",
                f"{results['score'].mean():.4f}",
                f"{results['score'].max() - results['score'].min():.4f}",
            ],
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

else:
    st.markdown("""
    <div class="empty-state">
        <div style="font-size:2.2rem; margin-bottom:0.6rem;">⚖️</div>
        <div class="empty-title">Enter a query to search the corpus</div>
        <p style="font-size:0.88rem; margin:0.3rem 0 0 0; color:#9CA3AF;">
            Type a legal concept, fact, rule, or phrase above — the engine ranks documents by semantic similarity.
        </p>
        <div class="empty-examples">
            <span class="example-chip">habeas corpus</span>
            <span class="example-chip">right to counsel</span>
            <span class="example-chip">custodial torture</span>
            <span class="example-chip">fundamental rights</span>
            <span class="example-chip">judicial review</span>
            <span class="example-chip">due process violation</span>
            <span class="example-chip">bail conditions</span>
            <span class="example-chip">contempt of court</span>
        </div> 
    </div>
    """, unsafe_allow_html=True)
