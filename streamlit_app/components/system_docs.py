"""
System documentation data artifacts and architectural specifications for RecSysLab.
Provides diagram strings, LaTeX mathematical formulas, prompt schemas, and API contracts.
"""

# ============================================================
# DIAGRAMS
# ============================================================

SYSTEM_HERO_DIAGRAM = """
+-----------------------------------------------------------------------------------+
|                                   USER INTERFACE                                  |
|               Streamlit Web App (Recommend / Evaluation / System)                 |
+-----------------------------------------+-----------------------------------------+
                                          |
                                    REST API (HTTP)
                                          |
+-----------------------------------------v-----------------------------------------+
|                                 FASTAPI BACKEND                                   |
|   /health Endpoint  |  /recommendations Endpoint  |  Pydantic Schemas & Lifespan   |
+-----------------------------------------+-----------------------------------------+
                                          |
                                    Service Layer
                                          |
+-----------------------------------------v-----------------------------------------+
|                             RECOMMENDATION SERVICE                                |
|                                                                                   |
|  +-----------------------+     +-----------------------+     +-----------------+  |
|  | Stage 1: Retrieval    | --> | Stage 2: Ranking      | --> | Stage 3: LLM    |  |
|  | BGE-M3 + FAISS Index  |     | 4-Factor Fusion + CE  |     | Gemini Flash    |  |
|  +-----------------------+     +-----------------------+     +-----------------+  |
+-----------------------------------------------------------------------------------+
"""

OFFLINE_PIPELINE_DIAGRAM = """
[MovieLens 32M Raw Data]
       │
       ├─► movies.csv   (movieId, title, genres)
       ├─► tags.csv     (movieId, tag, relevance)
       └─► ratings.csv  (movieId, rating, timestamp)
       │
       ▼
[Data Preprocessing Pipeline]
       │
       ├─► Filter movies with rating_count > 50
       ├─► Aggregate top 40 tags per movie
       ├─► Compute mean rating & rating count
       └─► Construct rich metadata `page_content`
       │
       ▼
[Processed Parquet Dataset]
       │  (movies_processed.parquet)
       ▼
[Embedding Generation (BAAI/bge-m3)]
       │  (Dense vector embeddings with L2 normalization)
       ▼
[FAISS Vector Store Index]
       │  (faiss_movie_index_refactored)
       ▼
[Ready for Real-time Retrieval]
"""

ONLINE_PIPELINE_DIAGRAM = """
[User Natural-Language Query] (e.g. "sad movie about father and son")
       │
       ▼
[1. Candidate Retrieval (FAISS)]
       │  Vectorize query via BAAI/bge-m3 ──► Top 20 Candidates
       ▼
[2. Multi-Factor Score Calculation]
       │
       ├─► Semantic Score    : Normalized similarity from FAISS distance (25%)
       ├─► Popularity Score  : rating_mean * log(1 + rating_count) (15%)
       ├─► Rule-based Score  : Genre boost/penalty keyword mapping (10%)
       └─► Cross-Encoder     : BAAI/bge-reranker-v2-m3 pairwise scoring (50%)
       │
       ▼
[3. Weighted Fusion & Sorting]
       │  Compute Final Score ──► Sort descending ──► Top-N Movies
       ▼
[4. LLM Explanation Generation (Gemini 3.5 Flash Lite)]
       │  Format Top-N context ──► Prompt ──► Structured JSON Explanations
       ▼
[Recommendation Response Payload] (Movies + Scores + Explanations + Timings)
"""


# ============================================================
# RANKING FORMULA & WEIGHTS
# ============================================================

RANKING_FORMULA_LATEX = r"""
\text{Final Score} = \alpha \cdot S_{\text{Cross-Encoder}} + \beta \cdot S_{\text{Semantic}} + \gamma \cdot S_{\text{Popularity}} + \delta \cdot S_{\text{Rule}}
"""

RANKING_FORMULA_VALUES_LATEX = r"""
\text{Final Score} = 0.50 \cdot S_{\text{CE}} + 0.25 \cdot S_{\text{Semantic}} + 0.15 \cdot S_{\text{Popularity}} + 0.10 \cdot S_{\text{Rule}}
"""

WEIGHT_DESCRIPTIONS = [
    {
        "factor": "Cross-Encoder Score",
        "weight": "50% (α = 0.50)",
        "model": "BAAI/bge-reranker-v2-m3",
        "description": "Đánh giá mối quan hệ tương quan ngữ nghĩa sâu từng cặp (query, document). Đóng vai trò quan trọng nhất trong việc bắt được tinh tế cảm xúc và đa điều kiện.",
    },
    {
        "factor": "Semantic Similarity",
        "weight": "25% (β = 0.25)",
        "model": "BAAI/bge-m3 + FAISS",
        "description": "Đo khoảng cách Vector giữa query và metadata phim trong không gian biểu diễn đa chiều. Đảm bảo giữ được ứng viên phù hợp ngữ cảnh gốc.",
    },
    {
        "factor": "Popularity Score",
        "weight": "15% (γ = 0.15)",
        "model": "Mean Rating × log(1 + Count)",
        "description": "Kết hợp điểm đánh giá trung bình và số lượng đánh giá của cộng đồng để ưu tiên các bộ phim chất lượng cao được công nhận.",
    },
    {
        "factor": "Rule-based Score",
        "weight": "10% (δ = 0.10)",
        "model": "Genre Boost / Penalty Mapping",
        "description": "Quy tắc từ khóa nhẹ giúp tăng/giảm điểm theo thể loại (ví dụ: 'sad' tăng điểm Drama, trừ điểm Comedy/Horror). Rất hiệu quả cho truy vấn phủ định (negative constraint).",
    },
]


# ============================================================
# LLM PROMPT ARCHITECTURE
# ============================================================

LLM_PROMPT_ARCH = {
    "model": "Google Gemini 3.5 Flash Lite",
    "temperature": 0.2,
    "system_prompt": """You are an expert movie recommendation assistant.
Your task is to explain why each ranked movie matches the user's request.

Strict rules:
1. ONLY discuss movies included in the provided context.
2. NEVER invent or recommend any additional movie.
3. NEVER change the ranking order.
4. Create exactly ONE explanation for EACH ranked movie.
5. Base every explanation ONLY on: genres, rating information, semantic metadata.
6. Write explanations in Vietnamese within 2 sentences (max 60 words per explanation).
7. Return ONLY valid JSON array adhering to the output schema.""",
    "context_format": """Rank {rank}
Title: {title}
Genres: {genres}
Rating: {rating_mean}/5.00 from {rating_count} ratings
Semantic metadata:
{page_content}""",
    "json_schema": """[
  {
    "rank": 1,
    "explanation": "Lời giải thích bằng tiếng Việt tối đa 2 câu..."
  }
]""",
}


# ============================================================
# API CONTRACT SAMPLES
# ============================================================

API_HEALTH_SAMPLE = """{
  "status": "healthy",
  "service": "RecSysLab Movie Recommendation API",
  "version": "1.0.0",
  "vector_store_loaded": true,
  "default_mode": "quality"
}"""

API_RECOMMEND_REQUEST_SAMPLE = """{
  "query": "sad movie about father and son",
  "mode": "quality",
  "top_k": 5,
  "include_explanation": true
}"""

API_RECOMMEND_RESPONSE_SAMPLE = """{
  "query": "sad movie about father and son",
  "mode": "quality",
  "configuration": "hybrid_with_ce",
  "top_k": 5,
  "recommendations": [
    {
      "final_rank": 1,
      "movieId": 318,
      "title": "Shawshank Redemption, The (1994)",
      "genres": "Crime|Drama",
      "rating_mean": 4.41,
      "rating_count": 122296,
      "final_score": 0.8942,
      "explanation": "Bộ phim kịch tính sâu sắc về hy vọng và tình cảm gia đình, đạt đánh giá 4.41/5 xuất sắc."
    }
  ],
  "timings": {
    "ranking_seconds": 0.1852,
    "generation_seconds": 0.8410,
    "total_seconds": 1.0262
  }
}"""


# ============================================================
# CODEBASE TREE STRUCTURE
# ============================================================

CODEBASE_TREE = """
RecSysLab/
├── data/
│   ├── configs/            # Config yaml (production.yaml)
│   ├── evaluation/         # Test queries, predictions, ground truth labels
│   ├── faiss_movie_index/  # Dense vector store index (FAISS)
│   └── src/
│       ├── api/            # FastAPI routes, dependencies, Pydantic schemas
│       │   ├── routes/     # /health, /recommendations endpoints
│       │   └── app.py      # App lifespan & FastAPI startup
│       ├── data/           # Preprocessing & document creation
│       ├── evaluation/     # Ranking configs & NDCG/MRR metrics
│       ├── generation/     # LangChain explanation chain & prompts
│       ├── ranking/        # Cross-Encoder & 4-factor Hybrid Ranker
│       ├── retrieval/      # BGE-M3 embeddings & FAISS vector store
│       └── services/       # RecommendationService orchestrator
├── docs/                   # Benchmark evaluation summary docs
├── notebooks/              # EDA, prototype scripts & pipeline tests
└── streamlit_app/          # Streamlit Multi-Page Web Interface
    ├── app.py              # Main dashboard overview
    ├── components/         # Reusable UI components & docs
    │   ├── evaluation/     # Case study & benchmark charts
    │   └── system_docs.py  # System documentation artifacts
    ├── pages/              # Multi-page views
    │   ├── 1_Recommend.py  # Interactive recommendation page
    │   ├── 2_Evaluation.py # Evaluation benchmark dashboard
    │   └── 3_System.py     # System Architecture documentation page
    ├── services/           # REST API client
    └── utils/              # Theme, config, and data helpers
"""
