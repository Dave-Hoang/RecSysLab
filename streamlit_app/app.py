import streamlit as st
from utils.theme import load_theme
from services.api_client import api_client

st.set_page_config(
    page_title="RecSysLab",
    page_icon="🎬",
    layout="wide",
)

load_theme()

st.title("🎬 RecSysLab")

st.markdown("""
## Hybrid Semantic Movie Recommendation System

**Applied AI Engineer Portfolio**

Built with **Hybrid Retrieval**, **Cross Encoder Reranking**, and
**LLM-powered Explanation Generation** to deliver accurate and explainable
movie recommendations.
""")

st.divider()

st.subheader("✨ Key Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
- Semantic Movie Search
- Hybrid Retrieval
- Cross Encoder Reranking
""")

with col2:
    st.markdown("""
- Explainable Recommendation
- FastAPI Inference API
- Interactive Streamlit Demo
""")

st.divider()

st.subheader("🏗 Technology Stack")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🤖 AI & Machine Learning")
    st.markdown("""
- PyTorch
- Sentence Transformers
- BGE-M3
- Cross Encoder
- Google Gemini
""")

    st.markdown("### 🔍 Retrieval")
    st.markdown("""
- FAISS
- Hybrid Retrieval
- Dense Vector Search
""")

with col2:
    st.markdown("### ⚙ Backend")
    st.markdown("""
- FastAPI
- Pydantic
- Uvicorn
""")

    st.markdown("### 🎨 Frontend")
    st.markdown("""
- Streamlit
- Custom CSS
- Python
""")

st.divider()

st.subheader("⚡ System Status")

try:
    status = api_client.health()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Backend",
            "🟢 Online",
        )

    with col2:
        loaded = "Loaded" if status.get("vector_store_loaded") else "Not Loaded"
        st.metric(
            "Vector Store",
            loaded,
        )

    with col3:
        st.metric(
            "Default Mode",
            status.get("default_mode", "-"),
        )

    with col4:
        st.metric(
            "LLM",
            "Gemini",
        )

except Exception:
    st.error("Backend is currently unavailable.")

st.divider()

st.subheader("🏗 Recommendation Pipeline")

st.markdown("""
```text
User Query
      │
      ▼
BGE-M3 Embedding
      │
      ▼
FAISS Retrieval
      │
      ▼
Hybrid Ranking
      │
      ▼
Cross Encoder Reranking
      │
      ▼
Gemini Explanation
      │
      ▼
Top-K Recommendations
```
""")
