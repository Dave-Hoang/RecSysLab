import streamlit as st

from components import system_docs
from services.api_client import api_client
from utils.theme import load_theme

# Set page configuration
st.set_page_config(
    page_title="System Architecture - RecSysLab",
    page_icon="⚙️",
    layout="wide",
)

# Load custom theme CSS
load_theme()

# Title and subtitle
st.title("⚙️ System Architecture")
st.caption(
    "Technical documentation and architectural design of RecSysLab recommendation engine."
)

st.divider()

# ============================================================
# LIVE SYSTEM STATUS BAR
# ============================================================
st.subheader("⚡ Live System Status")

try:
    health_data = api_client.health()
    backend_status = "🟢 Online"
    vector_store_status = (
        "🟢 Loaded"
        if health_data.get("vector_store_loaded")
        else "🔴 Not Loaded"
    )
    default_mode = health_data.get("default_mode", "quality").upper()
    llm_model = "Gemini 2.5 Flash"
except Exception:
    backend_status = "🔴 Offline"
    vector_store_status = "⚠️ Unknown"
    default_mode = "QUALITY"
    llm_model = "Gemini 2.5 Flash"

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    st.metric("FastAPI Backend", backend_status)

with col_s2:
    st.metric("FAISS Vector Index", vector_store_status)

with col_s3:
    st.metric("Production Mode", default_mode)

with col_s4:
    st.metric("LLM Provider", llm_model)

st.divider()

# ============================================================
# SECTION 1: SYSTEM OVERVIEW & TECH STACK
# ============================================================
st.subheader("🏛 System Overview & Tech Stack")

with st.container(border=True):
    st.markdown("##### 📌 End-to-End System Flow")
    st.code(system_docs.SYSTEM_HERO_DIAGRAM, language="text")

st.markdown("#### 🏗 Technology Stack Breakdown")

col_t1, col_t2, col_t3 = st.columns(3)

with col_t1:
    with st.container(border=True):
        st.markdown("### 🤖 AI & Machine Learning")
        st.markdown("""
- **Embedding Model**: `BAAI/bge-m3`
- **Reranker**: `BAAI/bge-reranker-v2-m3`
- **LLM Engine**: `Google Gemini 2.5 Flash`
- **Framework**: `PyTorch` & `LangChain`
""")

with col_t2:
    with st.container(border=True):
        st.markdown("### 🔍 Vector Search & Retrieval")
        st.markdown("""
- **Vector DB**: `FAISS` (Facebook AI Similarity Search)
- **Vector Metric**: Normalized L2 / Inner Product
- **Index Type**: IndexFlatL2 (Refactored)
- **Candidate Pool**: Top-20 Candidates
""")

with col_t3:
    with st.container(border=True):
        st.markdown("### ⚙ Backend & Engineering")
        st.markdown("""
- **REST API**: `FastAPI` + `Uvicorn`
- **Schema Validation**: `Pydantic v2`
- **Frontend**: `Streamlit`
- **Data Storage**: `Parquet` & `Pandas`
""")

st.divider()

# ============================================================
# SECTION 2 & 3: OFFLINE DATA VS ONLINE INFERENCE PIPELINES
# ============================================================
st.subheader("🔄 Data & Inference Pipelines")

tab_offline, tab_online = st.tabs(
    ["📦 Offline Data Pipeline", "⚡ Real-Time Online Pipeline"]
)

with tab_offline:
    st.markdown("#### 📦 Offline Data Processing & Vector Indexing")
    
    with st.container(border=True):
        st.code(system_docs.OFFLINE_PIPELINE_DIAGRAM, language="text")
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        with st.container(border=True):
            st.markdown("##### Step 1: Preprocessing & Filtering")
            st.markdown("""
- Lọc các phim có số lượng đánh giá `rating_count > 50` từ MovieLens 32M.
- Tổng hợp 40 tags phổ biến nhất per-movie.
- Tính toán điểm đánh giá trung bình `rating_mean`.
""")
    
    with col_p2:
        with st.container(border=True):
            st.markdown("##### Step 2: Dense Vector Indexing")
            st.markdown("""
- Tạo chuỗi metadata phong phú `page_content` cho từng phim.
- Sinh vector nhúng 1024 chiều bằng model `BAAI/bge-m3`.
- NORMALIZE embeddings và lưu trữ vào FAISS Index đĩa.
""")

with tab_online:
    st.markdown("#### ⚡ Real-Time Online Recommendation Pipeline")
    
    with st.container(border=True):
        st.code(system_docs.ONLINE_PIPELINE_DIAGRAM, language="text")
    
    col_o1, col_o2, col_o3 = st.columns(3)
    
    with col_o1:
        with st.container(border=True):
            st.markdown("##### Stage 1: Retrieval")
            st.markdown("""
- Nhận query tiếng Việt / tiếng Anh từ User.
- Embedding query bằng `bge-m3`.
- FAISS search lấy Top-20 candidates gần nhất.
""")
    
    with col_o2:
        with st.container(border=True):
            st.markdown("##### Stage 2: Hybrid Reranking")
            st.markdown("""
- Tính 4 điểm thành phần (Cross-Encoder, Semantic, Popularity, Rules).
- Xếp hạng theo trọng số Weighted Fusion.
- Trả về Top-N phim xuất sắc nhất.
""")

    with col_o3:
        with st.container(border=True):
            st.markdown("##### Stage 3: LLM Explanation")
            st.markdown("""
- Định dạng ngữ cảnh Top-N phim.
- Gửi Prompt tới Google Gemini Flash.
- Nhận lời giải thích tiếng Việt dạng JSON chuẩn.
""")

st.divider()

# ============================================================
# SECTION 4: RANKING PIPELINE & WEIGHTED FUSION MATH
# ============================================================
st.subheader("🧮 Multi-Factor Ranking & Weighted Fusion")

st.markdown("""
Hệ thống kết hợp 4 nguồn tín hiệu điểm số để đảm bảo độ chính xác ngữ nghĩa, độ phổ biến của phim và khớp thể loại:
""")

with st.container(border=True):
    st.latex(system_docs.RANKING_FORMULA_LATEX)
    st.latex(system_docs.RANKING_FORMULA_VALUES_LATEX)

with st.expander("📖 Chi Tiết Ý Nghĩa Các Trọng Số Scoring Factor", expanded=True):
    for weight_info in system_docs.WEIGHT_DESCRIPTIONS:
        st.markdown(
            f"**• {weight_info['factor']} ({weight_info['weight']})** — *Model/Logic: `{weight_info['model']}`*"
        )
        st.markdown(f"> {weight_info['description']}")
        st.markdown("")

st.divider()

# ============================================================
# SECTION 5: LLM EXPLANATION & PROMPT ARCHITECTURE
# ============================================================
st.subheader("💡 LLM Explanation Chain & Prompt Architecture")

col_l1, col_l2 = st.columns([1, 1])

with col_l1:
    with st.container(border=True):
        st.markdown("##### 📝 System Prompt Constraints")
        st.code(system_docs.LLM_PROMPT_ARCH["system_prompt"], language="text")

with col_l2:
    with st.container(border=True):
        st.markdown("##### 📄 Context Format per Movie")
        st.code(system_docs.LLM_PROMPT_ARCH["context_format"], language="text")
        
        st.markdown("##### 🎯 Expected Output JSON Schema")
        st.code(system_docs.LLM_PROMPT_ARCH["json_schema"], language="json")

st.divider()

# ============================================================
# SECTION 6: API SPECIFICATIONS & CONTRACTS
# ============================================================
st.subheader("🔌 API Specifications (Swagger Contract)")

tab_health, tab_recommend = st.tabs(
    ["GET /health", "POST /recommendations"]
)

with tab_health:
    st.markdown("#### `GET /health` — Check Backend & Resource Status")
    st.code(system_docs.API_HEALTH_SAMPLE, language="json")

with tab_recommend:
    st.markdown("#### `POST /recommendations` — Movie Recommendation Endpoint")
    col_req, col_res = st.columns(2)
    
    with col_req:
        st.markdown("##### Request Payload Sample")
        st.code(system_docs.API_RECOMMEND_REQUEST_SAMPLE, language="json")
    
    with col_res:
        st.markdown("##### Response Payload Sample")
        st.code(system_docs.API_RECOMMEND_RESPONSE_SAMPLE, language="json")

st.divider()

# ============================================================
# SECTION 7: CLEAN CODEBASE ARCHITECTURE
# ============================================================
st.subheader("📂 Codebase Directory Architecture")

with st.container(border=True):
    st.code(system_docs.CODEBASE_TREE, language="text")
