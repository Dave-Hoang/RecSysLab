# RecSysLab: Production-Grade Hybrid Semantic Recommendation & Reranking Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blue?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-3.5_Flash_Lite-8E75B2?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

> **Applied AI Engineer Portfolio Project**
> An end-to-end, two-stage hybrid semantic movie recommendation system featuring **Dense Retrieval (FAISS + BGE-M3)**, **4-Factor Multi-Modal Weighted Fusion Reranking (Cross-Encoder)**, and **LLM-Powered Natural Language Explanation Generation (Gemini 3.5 Flash Lite)**.

---

## Performance & Latency Benchmarks

Designed with production latency SLA constraints in mind, separating low-latency candidate retrieval & reranking from real-time LLM explanation generation.

```
End-to-End Pipeline Latency Budget
┌─────────────────────────────────────────────────────────────┬───────────────────────────────┐
│ Stage                                                       │ Latency (Avg)                 │
├─────────────────────────────────────────────────────────────┼───────────────────────────────┤
│ 1. Dense Vector Candidate Retrieval (FAISS + BGE-M3)        │ ~0.085s                       │
│ 2. 4-Factor Multi-Signal Scoring & Reranking (Cross-Encoder)│ ~0.685s                       │
│ SUB-SECOND RANKING TOTAL (Fast Mode)                        │ 0.770s (Sub-second SLA)       │
├─────────────────────────────────────────────────────────────┼───────────────────────────────┤
│ 3. Natural Language Explanation Generation (Gemini 3.5 Lite)│ ~2.456s                       │
│ END-TO-END TOTAL (Quality Mode)                             │ 3.225s                        │
└─────────────────────────────────────────────────────────────┴───────────────────────────────┘
```

### Mode Latency Comparison

| Mode | Target SLA | Latency | Use Case | Included Components |
| :--- | :---: | :---: | :--- | :--- |
| **Fast Mode** | `< 1.0s` | **`0.770s`** | High-throughput APIs, mobile endpoints, batch recommendations | FAISS Retrieval + Hybrid Reranking |
| **Quality Mode** | `< 5.0s` | **`3.225s`** | Interactive search dashboard, personalized explainable recommendations | FAISS + Hybrid Reranking + LLM Explanation |

---

## Offline Evaluation & Accuracy Impact

Evaluated on a benchmark suite of **30 human-labeled test queries** across **6 semantic categories** (*Emotion/Theme, Genre, Multi-condition, Natural Language, Negative Constraint, Similar Movie*) and **3 difficulty levels** (*Easy, Medium, Hard*).

### Key Benchmark Highlights

* **+13.0% NDCG@5 Improvement**: Increased NDCG@5 from `0.733` (FAISS baseline) to **`0.828`** (Hybrid + Cross Encoder).
* **91.3% Precision@5**: Achieved **`0.913`** mean precision@5 across all test query categories.
* **-38.1% Noise Reduction**: Decreased irrelevant candidates per top-5 results from `0.700` down to **`0.433`**.
* **95.0% MRR@5**: Mean Reciprocal Rank reached **`0.950`**, ensuring relevant items appear in top positions.

### 4-Pipeline Benchmark Matrix

| Pipeline Configuration | NDCG@5 Mean | Precision@5 | MRR@5 | Irrelevant@5 | Mean Relevance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hybrid + Cross Encoder** *(Production)* | **`0.828`** | **`0.913`** | **`0.950`** | **`0.433`** | **`1.627`** |
| **Hybrid (No Cross Encoder)** | `0.774` | `0.880` | `0.967` | `0.600` | `1.513` |
| **Cross Encoder Only** | `0.768` | `0.873` | `0.911` | `0.633` | `1.520` |
| **FAISS Baseline (Dense Only)** | `0.733` | `0.860` | `0.928` | `0.700` | `1.473` |

---

## System Architecture

The recommendation engine adopts an industry-standard **Two-Stage Retrieval & Reranking Architecture** coupled with an async LLM Explanation layer.

```
                               ┌─────────────────────────────────────────┐
                               │           USER INTERFACE                │
                               │  Streamlit Multi-Page Web Dashboard     │
                               └────────────────────┬────────────────────┘
                                                    │ REST API (HTTP)
                               ┌────────────────────▼────────────────────┐
                               │            FASTAPI BACKEND              │
                               │  /health  |  /recommendations Endpoints │
                               └────────────────────┬────────────────────┘
                                                    │ Service Layer
  ┌─────────────────────────────────────────────────▼─────────────────────────────────────────────────┐
  │                                    RECOMMENDATION SERVICE                                         │
  │                                                                                                   │
  │  ┌───────────────────────┐        ┌───────────────────────┐        ┌───────────────────────────┐  │
  │  │ Stage 1: Retrieval    │ ────►  │ Stage 2: Reranking    │ ────►  │ Stage 3: LLM Explanation  │  │
  │  │ BGE-M3 + FAISS Index  │ Top-20 │ 4-Factor Weighted     │ Top-N  │ Google Gemini 3.5         │  │
  │  │ Candidate Selection   │        │ Fusion + Cross-Encoder│        │ Flash Lite                │  │
  │  └───────────────────────┘        └───────────────────────┘        └───────────────────────────┘  │
  └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Multi-Factor Weighted Fusion Formula

To balance deep semantic alignment, community popularity signals, and genre keyword rules, candidates are scored using a multi-factor weighted score fusion model:

$$\text{Final Score} = \alpha \cdot S_{\text{Cross-Encoder}} + \beta \cdot S_{\text{Semantic}} + \gamma \cdot S_{\text{Popularity}} + \delta \cdot S_{\text{Rule}}$$

Where:
* **$\alpha$**: Weight assigned to Cross-Encoder pairwise relevance score ($S_{\text{Cross-Encoder}}$), computed via `BAAI/bge-reranker-v2-m3`.
* **$\beta$**: Weight assigned to Dense Semantic similarity score ($S_{\text{Semantic}}$), derived from normalized cosine distance in 1024-dim embedding space (`BAAI/bge-m3` + FAISS).
* **$\gamma$**: Weight assigned to Movie Popularity score ($S_{\text{Popularity}}$), normalized via `rating_mean * log(1 + rating_count)`.
* **$\delta$**: Weight assigned to Rule-Based Genre score ($S_{\text{Rule}}$), heuristic keyword boosting/penalizing for constraint handling.

---

## Tech Stack & Key Design Choices

### Machine Learning & AI
* **Embedding Model**: `BAAI/bge-m3` (Dense multi-lingual vector representations)
* **Reranker**: `BAAI/bge-reranker-v2-m3` (State-of-the-art cross-encoder)
* **LLM Engine**: `Google Gemini 3.5 Flash Lite` (Strict JSON schema constrained generation)
* **Frameworks**: `PyTorch`, `LangChain`, `Sentence-Transformers`

### Vector Storage & Engineering
* **Vector Database**: `FAISS` (`IndexFlatL2` with unit L2 normalized vectors for fast cosine similarity)
* **Dataset**: MovieLens 32M (filtered for `rating_count > 50`, top 40 tags aggregated into rich metadata `page_content`)
* **Data Pipelines**: `Pandas`, `PyArrow`, `Parquet`

### Backend & Frontend Systems
* **Backend API**: `FastAPI`, `Uvicorn`, `Pydantic v2`
* **Web Interface**: `Streamlit` (Custom CSS, interactive benchmark analytics, live case study viewer)

---

## Codebase Architecture

```
RecSysLab/
├── data/
│   ├── configs/                # Production YAML configurations
│   ├── evaluation/             # Test queries, ground truth labels & benchmark outputs
│   │   └── results/            # CSV metric summaries (overall, category, difficulty)
│   ├── faiss_movie_index/      # Dense FAISS vector index artifacts
│   └── src/
│       ├── api/                # FastAPI application, routes & Pydantic schemas
│       ├── data/               # MovieLens data ingestion & text generation
│       ├── evaluation/         # Benchmark evaluation runner & NDCG/MRR calculators
│       ├── generation/         # Gemini LLM explanation chain & prompt templates
│       ├── ranking/            # 4-Factor Weighted Fusion & Cross-Encoder rankers
│       ├── retrieval/          # BGE-M3 embedding generator & FAISS retriever
│       └── services/           # RecommendationService orchestrator
├── docs/                       # Benchmark evaluation summaries
├── streamlit_app/              # Streamlit Multi-Page Web Interface
│   ├── app.py                  # Main overview dashboard
│   ├── components/             # Reusable UI components & docs
│   ├── pages/
│   │   ├── 1_Recommend.py      # Interactive search & recommendation page
│   │   ├── 2_Evaluation.py     # Offline evaluation & case study dashboard
│   │   └── 3_System.py         # System architecture & live API status page
│   ├── services/               # REST API client
│   └── utils/                  # Data loaders, theme & config utilities
└── README.md                   # System documentation & portfolio presentation
```

---

## Quick Start & Installation

### 1. Clone & Setup Environment

```bash
git clone https://github.com/Dave-Hoang/RecSysLab.git
cd RecSysLab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
API_BASE_URL=http://127.0.0.1:8000
```

### 3. Launch Backend API & Frontend Dashboard

```bash
# Start FastAPI Backend API (Port 8000)
uvicorn data.src.api.app:app --reload --port 8000

# Start Streamlit Web Dashboard (Port 8501)
streamlit run streamlit_app/app.py
```

Open `http://localhost:8501` in your browser to interact with the system!

---

## License & Contact

* **Author**: Applied AI Engineering Portfolio
* **License**: MIT License
