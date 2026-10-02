from collections.abc import Mapping

import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

import logging
logger = logging.getLogger(__name__)

from src.config import (
    FINAL_RECOMMENDATION_TOP_K,
    PRE_RANK_TOP_K,
    RETRIEVAL_TOP_K,
    RULE_SCORE_MAX,
    RULE_SCORE_MIN,
    RULE_SCORE_STEP,
    WEIGHT_CROSS_ENCODER,
    WEIGHT_POPULARITY,
    WEIGHT_RULE,
    WEIGHT_SEMANTIC,
    CROSS_ENCODER_RAW_LOGIT_FLOOR,
    CROSS_ENCODER_FALLBACK_MAX_SCORE,
)
from src.ranking.cross_encoder import predict_relevance_scores
from src.retrieval.retriever import retrieve_movies_with_score


DEFAULT_RULES: dict[str, dict[str, list[str]]] = {
    "sad": {
        "boost": ["Drama"],
        "penalty": ["Comedy", "Horror"],
    },
    "emotional": {
        "boost": ["Drama"],
        "penalty": ["Comedy", "Horror"],
    },
    "father": {
        "boost": ["Drama"],
        "penalty": ["Horror"],
    },
    "scary": {
        "boost": ["Horror", "Thriller"],
        "penalty": [],
    },
    "ghost": {
        "boost": ["Horror", "Thriller"],
        "penalty": [],
    },
    "romantic": {
        "boost": ["Romance", "Comedy"],
        "penalty": ["Horror"],
    },
}


REQUIRED_CANDIDATE_COLUMNS = {
    "rank",
    "movieId",
    "title",
    "genres",
    "rating_mean",
    "rating_count",
    "page_content",
}


def _validate_candidate_columns(
    candidates: pd.DataFrame,
    required_columns: set[str],
) -> None:
    """
    Kiểm tra candidate DataFrame có đủ cột cần thiết.
    """
    missing_columns = required_columns.difference(
        candidates.columns
    )

    if missing_columns:
        raise ValueError(
            "Candidate DataFrame thiếu các cột: "
            f"{sorted(missing_columns)}"
        )


def _min_max_normalize_np(
    values: np.ndarray,
    constant_value: float = 0.5,
) -> np.ndarray:
    """
    Min-max normalize mảng 1D NumPy về khoảng 0–1.

    Nếu toàn bộ giá trị bằng nhau, trả về constant_value cho tất cả phần tử.
    """
    numeric_values = np.nan_to_num(values, nan=0.0)
    if numeric_values.size == 0:
        return numeric_values

    minimum = float(np.min(numeric_values))
    maximum = float(np.max(numeric_values))

    value_range = maximum - minimum

    if value_range <= 0:
        return np.full_like(numeric_values, constant_value, dtype=np.float64)

    return (numeric_values - minimum) / value_range


def _min_max_normalize(
    values: pd.Series,
    constant_value: float = 0.5,
) -> pd.Series:
    """
    Min-max normalize một Series về khoảng 0–1.

    Nếu toàn bộ giá trị bằng nhau, trả về constant_value
    cho tất cả phần tử.
    """
    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    ).fillna(0.0)

    norm_arr = _min_max_normalize_np(
        numeric_values.to_numpy(dtype=np.float64),
        constant_value=constant_value,
    )
    return pd.Series(
        norm_arr,
        index=numeric_values.index,
        dtype="float64",
    )


def compute_semantic_similarity(
    candidates: pd.DataFrame,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Chuyển FAISS score/distance thành semantic similarity chuẩn hóa.

    Hỗ trợ cả:
    - IndexFlatIP / Cosine (faiss_similarity): raw_similarity = faiss_similarity
    - IndexFlatL2 (faiss_distance): raw_similarity = 1 - distance / 2

    Sau đó min-max normalize trong Top-K candidates.
    """
    if candidates.empty:
        return candidates if inplace else candidates.copy()

    has_sim = "faiss_similarity" in candidates.columns
    has_dist = "faiss_distance" in candidates.columns

    if not (has_sim or has_dist):
        raise ValueError("Candidate DataFrame phải có cột 'faiss_similarity' hoặc 'faiss_distance'")

    result = candidates if inplace else candidates.copy()

    if has_sim:
        scores = pd.to_numeric(result["faiss_similarity"], errors="coerce").to_numpy(dtype=np.float64, na_value=0.0)
        raw_similarity = scores
    else:
        distances = pd.to_numeric(result["faiss_distance"], errors="coerce").to_numpy(dtype=np.float64, na_value=0.0)
        result["faiss_squared_l2_distance"] = distances
        raw_similarity = 1.0 - (distances / 2.0)

    result["cosine_similarity_raw"] = raw_similarity

    normalized = _min_max_normalize_np(raw_similarity)
    result["semantic_score_relative"] = np.clip(normalized, 0.0, 1.0)

    return result


def compute_popularity_score(
    candidates: pd.DataFrame,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Tính popularity score từ rating_mean và rating_count.

    Công thức raw:

        rating_mean * log(1 + rating_count)

    Sau đó min-max normalize trong tập Top-K.
    """
    if candidates.empty:
        return candidates if inplace else candidates.copy()

    if not inplace:
        _validate_candidate_columns(
            candidates,
            {"rating_mean", "rating_count"},
        )

    result = candidates if inplace else candidates.copy()

    rating_mean = pd.to_numeric(
        result["rating_mean"],
        errors="coerce",
    ).to_numpy(dtype=np.float64, na_value=0.0)

    rating_count = pd.to_numeric(
        result["rating_count"],
        errors="coerce",
    ).to_numpy(dtype=np.float64, na_value=0.0)

    raw_popularity = (
        rating_mean
        * np.log1p(np.maximum(rating_count, 0.0))
    )

    normalized = _min_max_normalize_np(raw_popularity)
    result["popularity_score"] = np.clip(normalized, 0.0, 1.0)

    return result


def compute_rule_score(
    candidates: pd.DataFrame,
    query: str,
    rules: Mapping[str, Mapping[str, list[str]]] = DEFAULT_RULES,
    step: float = RULE_SCORE_STEP,
    minimum_score: float = RULE_SCORE_MIN,
    maximum_score: float = RULE_SCORE_MAX,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Tính rule score dựa trên keyword của query và genres phim.

    Rule chỉ đóng vai trò boost hoặc penalty nhẹ,
    không thay thế semantic và Cross-Encoder ranking.
    """
    if candidates.empty:
        return candidates if inplace else candidates.copy()

    if not inplace:
        _validate_candidate_columns(
            candidates,
            {"genres"},
        )

    cleaned_query = query.strip().lower()

    if not cleaned_query:
        raise ValueError("Query không được để trống.")

    active_boost_genres: set[str] = set()
    active_penalty_genres: set[str] = set()

    for keyword, rule in rules.items():
        if keyword.lower() not in cleaned_query:
            continue

        active_boost_genres.update(
            rule.get("boost", [])
        )
        active_penalty_genres.update(
            rule.get("penalty", [])
        )

    result = candidates if inplace else candidates.copy()
    scores = np.zeros(len(result), dtype=np.float64)

    if active_boost_genres or active_penalty_genres:
        # Vectorized one-hot encoding cho các genres phân tách bởi "|"
        genres_dummies = result["genres"].astype(str).str.get_dummies(sep="|")
        
        # Xóa khoảng trắng dư thừa ở tên cột nếu có
        genres_dummies.columns = genres_dummies.columns.str.strip()

        # Áp dụng điểm boost
        boost_cols = list(active_boost_genres.intersection(genres_dummies.columns))
        if boost_cols:
            scores += genres_dummies[boost_cols].sum(axis=1).to_numpy(dtype=np.float64) * step

        # Áp dụng điểm penalty
        penalty_cols = list(active_penalty_genres.intersection(genres_dummies.columns))
        if penalty_cols:
            scores -= genres_dummies[penalty_cols].sum(axis=1).to_numpy(dtype=np.float64) * step

    result["rule_score"] = np.clip(scores, minimum_score, maximum_score)

    return result


def compute_cross_encoder_score(
    candidates: pd.DataFrame,
    query: str,
    cross_encoder: CrossEncoder | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Chấm Cross-Encoder score cho từng candidate.
    """
    if candidates.empty:
        return candidates if inplace else candidates.copy()

    if not inplace:
        _validate_candidate_columns(
            candidates,
            {"page_content"},
        )

    result = candidates if inplace else candidates.copy()

    scores = predict_relevance_scores(
        query=query,
        documents=result["page_content"].astype(str).tolist(),
        cross_encoder=cross_encoder,
    )

    if len(scores) != len(result):
        raise RuntimeError(
            "Số Cross-Encoder scores không khớp "
            "số lượng candidates."
        )

    raw_scores_series = pd.Series(scores, index=result.index)
    
    if raw_scores_series.empty:
        normalized = raw_scores_series
    else:
        max_logit = float(raw_scores_series.max())
        
        if pd.isna(max_logit):
            normalized = raw_scores_series
        elif max_logit < CROSS_ENCODER_RAW_LOGIT_FLOOR:
            logger.warning(
                f"Cross-encoder fallback triggered: max raw logit = {max_logit:.4f} "
                f"(< {CROSS_ENCODER_RAW_LOGIT_FLOOR}). Batch considered low-confidence."
            )
            ranks = raw_scores_series.rank(method='average')
            if len(ranks) > 1:
                rank_normalized = (ranks - 1) / (len(ranks) - 1)
            else:
                rank_normalized = pd.Series(1.0, index=ranks.index)
                
            normalized = rank_normalized * CROSS_ENCODER_FALLBACK_MAX_SCORE
        else:
            normalized = _min_max_normalize(raw_scores_series)

    result["cross_encoder_score"] = (
        normalized.clip(0.0, 1.0)
        .astype("float64")
    )

    return result


def validate_ranking_weights(
    weight_cross_encoder: float,
    weight_semantic: float,
    weight_popularity: float,
    weight_rule: float,
) -> None:
    """
    Kiểm tra các trọng số final score.
    """
    weights = {
        "weight_cross_encoder": weight_cross_encoder,
        "weight_semantic": weight_semantic,
        "weight_popularity": weight_popularity,
        "weight_rule": weight_rule,
    }

    negative_weights = {
        name: value
        for name, value in weights.items()
        if value < 0
    }

    if negative_weights:
        raise ValueError(
            "Ranking weights không được âm: "
            f"{negative_weights}"
        )

    total_weight = sum(weights.values())

    if not np.isclose(total_weight, 1.0):
        raise ValueError(
            "Tổng ranking weights phải bằng 1.0. "
            f"Hiện tại: {total_weight:.6f}"
        )


# Weights cho Fast Path & Pre-rank filtering (3 factors, không có Cross-Encoder).
# Redistribute từ 4-factor weights gốc (CE=0.50, Sem=0.25, Pop=0.15, Rule=0.10),
# loại bỏ CE weight và normalize lại tổng = 1.0.
WEIGHT_SEMANTIC_NO_CE = 0.50
WEIGHT_POPULARITY_NO_CE = 0.30
WEIGHT_RULE_NO_CE = 0.20


def compute_final_score(
    candidates: pd.DataFrame,
    weight_cross_encoder: float = WEIGHT_CROSS_ENCODER,
    weight_semantic: float = WEIGHT_SEMANTIC,
    weight_popularity: float = WEIGHT_POPULARITY,
    weight_rule: float = WEIGHT_RULE,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Tính final score và sắp xếp lại candidates.

    Công thức mặc định:

        0.55 * cross_encoder_score
        + 0.25 * semantic_score_relative
        + 0.10 * popularity_score
        + 0.10 * rule_score
    """
    if candidates.empty:
        return candidates if inplace else candidates.copy()

    required_score_columns = {
        "cross_encoder_score",
        "semantic_score_relative",
        "popularity_score",
        "rule_score",
    }

    if not inplace:
        _validate_candidate_columns(
            candidates,
            required_score_columns,
        )

    validate_ranking_weights(
        weight_cross_encoder=weight_cross_encoder,
        weight_semantic=weight_semantic,
        weight_popularity=weight_popularity,
        weight_rule=weight_rule,
    )

    result = candidates if inplace else candidates.copy()

    ce_arr = result["cross_encoder_score"].to_numpy(dtype=np.float64)
    sem_arr = result["semantic_score_relative"].to_numpy(dtype=np.float64)
    pop_arr = result["popularity_score"].to_numpy(dtype=np.float64)
    rule_arr = result["rule_score"].to_numpy(dtype=np.float64)

    final_scores = (
        weight_cross_encoder * ce_arr
        + weight_semantic * sem_arr
        + weight_popularity * pop_arr
        + weight_rule * rule_arr
    )
    result["final_score"] = np.round(final_scores, 6)

    result = (
        result
        .sort_values(
            by="final_score",
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    result["final_rank"] = (
        np.arange(len(result)) + 1
    )

    return result


def rerank_candidates(
    candidates: pd.DataFrame,
    query: str,
    cross_encoder: CrossEncoder | None = None,
    pre_rank_top_k: int | None = PRE_RANK_TOP_K,
) -> pd.DataFrame:
    """
    Chạy toàn bộ ranking trên candidate DataFrame đã retrieve.

    Pipeline:

        faiss_distance
        → semantic_score_relative
        → popularity_score
        → rule_score
        → pre-rank filtering (nếu pre_rank_top_k được chỉ định)
        → cross_encoder_score
        → final_score
    """
    if candidates.empty:
        return candidates.copy()

    # 1. Single Entry Validation
    _validate_candidate_columns(
        candidates,
        REQUIRED_CANDIDATE_COLUMNS,
    )

    # 2. Single Entry Copy
    result = candidates.copy()

    # 3. Chấm điểm trực tiếp (inplace) bằng Pure NumPy
    compute_semantic_similarity(result, inplace=True)
    compute_popularity_score(result, inplace=True)
    compute_rule_score(
        result,
        query=query,
        inplace=True,
    )

    # Pre-rank filtering: Giữ lại Top pre_rank_top_k trước khi suy luận Cross-Encoder
    if pre_rank_top_k is not None and len(result) > pre_rank_top_k:
        sem_arr = result["semantic_score_relative"].to_numpy(dtype=np.float64)
        pop_arr = result["popularity_score"].to_numpy(dtype=np.float64)
        rule_arr = result["rule_score"].to_numpy(dtype=np.float64)

        pre_rank_score = (
            sem_arr * WEIGHT_SEMANTIC_NO_CE
            + pop_arr * WEIGHT_POPULARITY_NO_CE
            + rule_arr * WEIGHT_RULE_NO_CE
        )
        result["pre_rank_score"] = pre_rank_score
        result = (
            result
            .sort_values(
                by="pre_rank_score",
                ascending=False,
                kind="stable",
            )
            .head(pre_rank_top_k)
            .reset_index(drop=True)
        )

    compute_cross_encoder_score(
        result,
        query=query,
        cross_encoder=cross_encoder,
        inplace=True,
    )
    result = compute_final_score(result, inplace=True)

    # Làm tròn các cột điểm số một lần duy nhất ở output cuối cùng
    score_columns = [
        "cosine_similarity_raw",
        "semantic_score_relative",
        "popularity_score",
        "rule_score",
        "cross_encoder_score",
        "pre_rank_score",
        "final_score",
    ]
    for col in score_columns:
        if col in result.columns:
            result[col] = result[col].round(6)

    return result


def rank_movies(
    vector_store: FAISS,
    query: str,
    retrieval_k: int = RETRIEVAL_TOP_K,
    top_n: int = FINAL_RECOMMENDATION_TOP_K,
    cross_encoder: CrossEncoder | None = None,
    pre_rank_top_k: int | None = PRE_RANK_TOP_K,
) -> pd.DataFrame:
    """
    Pipeline hoàn chỉnh từ FAISS retrieval đến Top-N reranked movies.

    Args:
        vector_store:
            FAISS vector store đã load.
        query:
            Query của người dùng.
        retrieval_k:
            Số candidate lấy từ FAISS.
        top_n:
            Số kết quả cuối cùng.
        cross_encoder:
            Cross-Encoder tùy chọn.
        pre_rank_top_k:
            Số candidate tối đa giữ lại sau bước lọc thô.

    Returns:
        Top-N movies đã rerank.
    """
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Query không được để trống.")

    if retrieval_k <= 0:
        raise ValueError("retrieval_k phải lớn hơn 0.")

    if top_n <= 0:
        raise ValueError("top_n phải lớn hơn 0.")

    if top_n > retrieval_k:
        raise ValueError(
            "top_n không được lớn hơn retrieval_k."
        )

    effective_pre_rank_k = (
        max(pre_rank_top_k, top_n)
        if pre_rank_top_k is not None
        else None
    )

    candidates = retrieve_movies_with_score(
        vector_store=vector_store,
        query=cleaned_query,
        k=retrieval_k,
    )

    candidate_frame = pd.DataFrame(candidates)

    ranked_frame = rerank_candidates(
        candidates=candidate_frame,
        query=cleaned_query,
        cross_encoder=cross_encoder,
        pre_rank_top_k=effective_pre_rank_k,
    )

    return ranked_frame.head(top_n).reset_index(drop=True)


# ============================================================
# LANGGRAPH HELPER FUNCTIONS
# ============================================================


def rank_without_ce(
    candidates_df: pd.DataFrame,
    query: str,
) -> pd.DataFrame:
    """
    Fast Path: Semantic + Popularity + Rule scoring only (NO Cross-Encoder).

    Dùng cho LangGraph Node 2 (Fast FAISS Retrieval) khi query đủ rõ ràng
    và không cần Cross-Encoder reranking nặng.

    Args:
        candidates_df:
            DataFrame từ FAISS retrieval (phải có các cột trong
            REQUIRED_CANDIDATE_COLUMNS).
        query:
            User query string.

    Returns:
        DataFrame đã xếp hạng với final_score (KHÔNG có cột
        cross_encoder_score), sắp xếp theo final_score giảm dần.
    """
    if candidates_df.empty:
        return candidates_df.copy()

    # 1. Single Entry Validation
    _validate_candidate_columns(
        candidates_df,
        REQUIRED_CANDIDATE_COLUMNS,
    )

    # 2. Single Entry Copy
    result = candidates_df.copy()

    # 3. Chấm điểm trực tiếp (inplace) bằng Pure NumPy
    compute_semantic_similarity(result, inplace=True)
    compute_popularity_score(result, inplace=True)
    compute_rule_score(result, query=query, inplace=True)

    # Tính final_score với 3 factors (redistribute weights) bằng Pure NumPy
    sem_arr = result["semantic_score_relative"].to_numpy(dtype=np.float64)
    pop_arr = result["popularity_score"].to_numpy(dtype=np.float64)
    rule_arr = result["rule_score"].to_numpy(dtype=np.float64)

    final_scores = (
        sem_arr * WEIGHT_SEMANTIC_NO_CE
        + pop_arr * WEIGHT_POPULARITY_NO_CE
        + rule_arr * WEIGHT_RULE_NO_CE
    )
    result["final_score"] = np.round(final_scores, 6)

    result = (
        result
        .sort_values(
            by="final_score",
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    result["final_rank"] = (
        np.arange(len(result)) + 1
    )

    # Làm tròn các cột điểm số một lần duy nhất ở output cuối cùng
    score_columns = [
        "cosine_similarity_raw",
        "semantic_score_relative",
        "popularity_score",
        "rule_score",
        "final_score",
    ]
    for col in score_columns:
        if col in result.columns:
            result[col] = result[col].round(6)

    return result


def rank_with_ce(
    candidates_df: pd.DataFrame,
    query: str,
    cross_encoder: CrossEncoder | None = None,
    pre_rank_top_k: int | None = PRE_RANK_TOP_K,
) -> pd.DataFrame:
    """
    Quality Path: Full 4-factor scoring (Semantic + Popularity + Rule + Cross-Encoder)
    kết hợp Pre-ranking (lọc thô) tối ưu hiệu năng.

    Wrapper quanh rerank_candidates() hiện có, dùng cho LangGraph Node 5
    (Full Hybrid + CE Reranking) khi query cần chất lượng cao nhất.

    Args:
        candidates_df:
            DataFrame từ FAISS retrieval.
        query:
            User query string (expanded query hoặc original query).
        cross_encoder:
            Cross-Encoder model instance.
        pre_rank_top_k:
            Số lượng candidate tối đa giữ lại sau bước lọc thô để đưa vào Cross-Encoder.

    Returns:
        DataFrame đã xếp hạng đầy đủ bao gồm cross_encoder_score.
    """
    return rerank_candidates(
        candidates=candidates_df,
        query=query,
        cross_encoder=cross_encoder,
        pre_rank_top_k=pre_rank_top_k,
    )