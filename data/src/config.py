from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# config.py nằm tại:
# project_root/src/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MOVIELENS_DIR = PROJECT_ROOT / "ml-32m"

MOVIES_PATH = MOVIELENS_DIR / "movies.csv"
TAGS_PATH = MOVIELENS_DIR / "tags.csv"
RATINGS_PATH = MOVIELENS_DIR / "ratings.csv"

PROCESSED_MOVIES_PATH = MOVIELENS_DIR / "movies_processed.parquet"

FAISS_INDEX_DIR = PROJECT_ROOT / "faiss_movie_index"
FAISS_REFACTORED_INDEX_DIR = PROJECT_ROOT / "faiss_movie_index_refactored"

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
RETRIEVAL_RESULTS_PATH = EVALUATION_DIR / "retrieval_test_results.csv"


# ============================================================
# DATA PREPROCESSING CONFIG
# ============================================================

TAG_TOP_K = 40
TAG_MIN_COUNT = 2

# Giữ các phim có nhiều hơn 50 lượt đánh giá,
# giống logic cũ: rating_count > 50
MIN_RATING_COUNT = 50


# ============================================================
# EMBEDDING AND RETRIEVAL CONFIG
# ============================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

NORMALIZE_EMBEDDINGS = True

RETRIEVAL_TOP_K = 20
FINAL_RECOMMENDATION_TOP_K = 5


# ============================================================
# HELPERS
# ============================================================

def ensure_project_directories() -> None:
    MOVIELENS_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    # Các thư mục cũ của bạn
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVALUATION_METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Thêm thư mục cấu hình production
    CONFIGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def validate_raw_data_paths() -> None:
    """
    Kiểm tra các file MovieLens đầu vào có tồn tại hay không.

    Raises:
        FileNotFoundError: Nếu thiếu ít nhất một file đầu vào.
    """
    required_paths = [
        MOVIES_PATH,
        TAGS_PATH,
        RATINGS_PATH,
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        missing_text = "\n".join(f"- {path}" for path in missing_paths)

        raise FileNotFoundError(
            "Không tìm thấy các file dữ liệu MovieLens sau:\n"
            f"{missing_text}"
        )
    

# ============================================================
# RANKING CONFIG
# ============================================================

CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

RULE_SCORE_STEP = 0.10
RULE_SCORE_MIN = -0.30
RULE_SCORE_MAX = 0.30

WEIGHT_CROSS_ENCODER = 0.50
WEIGHT_SEMANTIC = 0.25
WEIGHT_POPULARITY = 0.15
WEIGHT_RULE = 0.10

# ============================================================
# LLM / GENERATION CONFIG
# ============================================================

GEMINI_MODEL_NAME = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.2

GOOGLE_API_KEY_ENV_NAME = "GOOGLE_API_KEY"

ENV_PATH = PROJECT_ROOT.parent / ".env"

# ============================================================
# EVALUATION CONFIG
# ============================================================
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

# ------------------------------------------------------------
# INPUT DATA
# ------------------------------------------------------------

EVALUATION_QUERIES_PATH = (
    EVALUATION_DIR / "queries.csv"
)

EVALUATION_PREDICTIONS_PATH = (
    EVALUATION_DIR / "predictions.csv"
)

EVALUATION_CANDIDATES_FOR_LABELING_PATH = (
    EVALUATION_DIR / "candidates_for_labeling.csv"
)

EVALUATION_LABELS_PATH = (
    EVALUATION_DIR / "labels.csv"
)

EVALUATION_SCORED_PREDICTIONS_PATH = (
    EVALUATION_DIR / "scored_predictions.csv"
)

EVALUATION_LABELS_DIR = (
    EVALUATION_DIR / "labels"
)

# ------------------------------------------------------------
# LEGACY / PREVIOUS EVALUATION OUTPUTS
# ------------------------------------------------------------

EVALUATION_RESULTS_BY_QUERY_PATH = (
    EVALUATION_DIR / "results_by_query.csv"
)

EVALUATION_SUMMARY_PATH = (
    EVALUATION_DIR / "summary.csv"
)

EVALUATION_RESULTS_PATH = (
    EVALUATION_RESULTS_BY_QUERY_PATH
)

# ------------------------------------------------------------
# STAGE 4 METRIC OUTPUTS
# ------------------------------------------------------------

EVALUATION_METRICS_DIR = (
    EVALUATION_DIR / "results"
)

EVALUATION_PER_QUERY_METRICS_PATH = (
    EVALUATION_METRICS_DIR / "per_query_metrics.csv"
)

EVALUATION_OVERALL_METRICS_PATH = (
    EVALUATION_METRICS_DIR / "overall_metrics.csv"
)

EVALUATION_CATEGORY_METRICS_PATH = (
    EVALUATION_METRICS_DIR / "category_metrics.csv"
)

EVALUATION_DIFFICULTY_METRICS_PATH = (
    EVALUATION_METRICS_DIR / "difficulty_metrics.csv"
)

# ============================================================
# PRODUCTION CONFIGURATION
# ============================================================

CONFIGS_DIR = PROJECT_ROOT / "configs"

PRODUCTION_CONFIG_PATH = (
    CONFIGS_DIR / "production.yaml"
)