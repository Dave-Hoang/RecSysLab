from collections.abc import Mapping

import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

from src.config import (
    FINAL_RECOMMENDATION_TOP_K,
    RETRIEVAL_TOP_K,
    RULE_SCORE_MAX,
    RULE_SCORE_MIN,
    RULE_SCORE_STEP,
    WEIGHT_CROSS_ENCODER,
    WEIGHT_POPULARITY,
    WEIGHT_RULE,
    WEIGHT_SEMANTIC,
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
    "faiss_distance",
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


def _min_max_normalize(
    values: pd.Series,
    constant_value: float = 1.0,
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

    minimum = float(numeric_values.min())
    maximum = float(numeric_values.max())

    value_range = maximum - minimum

    if value_range <= 0:
        return pd.Series(
            constant_value,
            index=numeric_values.index,
            dtype="float64",
        )

    return (numeric_values - minimum) / value_range


def compute_semantic_similarity(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Chuyển FAISS distance thành semantic similarity chuẩn hóa.

    Với normalized embeddings và squared L2 distance:

        raw_similarity = 1 - distance / 2

    Sau đó min-max normalize trong Top-K candidates.

    Args:
        candidates:
            Candidate DataFrame có cột faiss_distance.

    Returns:
        DataFrame có thêm semantic_similarity.
    """
    if candidates.empty:
        return candidates.copy()

    _validate_candidate_columns(
        candidates,
        {"faiss_distance"},
    )

    result = candidates.copy()

    distances = pd.to_numeric(
        result["faiss_distance"],
        errors="coerce",
    ).fillna(0.0)

    raw_similarity = 1.0 - (distances / 2.0)

    result["semantic_similarity"] = (
        _min_max_normalize(raw_similarity)
        .clip(0.0, 1.0)
        .round(6)
    )

    return result


def compute_popularity_score(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tính popularity score từ rating_mean và rating_count.

    Công thức raw:

        rating_mean * log(1 + rating_count)

    Sau đó min-max normalize trong tập Top-K.
    """
    if candidates.empty:
        return candidates.copy()

    _validate_candidate_columns(
        candidates,
        {"rating_mean", "rating_count"},
    )

    result = candidates.copy()

    rating_mean = pd.to_numeric(
        result["rating_mean"],
        errors="coerce",
    ).fillna(0.0)

    rating_count = pd.to_numeric(
        result["rating_count"],
        errors="coerce",
    ).fillna(0.0)

    raw_popularity = (
        rating_mean
        * np.log1p(rating_count.clip(lower=0.0))
    )

    result["popularity_score"] = (
        _min_max_normalize(raw_popularity)
        .clip(0.0, 1.0)
        .round(6)
    )

    return result


def compute_rule_score(
    candidates: pd.DataFrame,
    query: str,
    rules: Mapping[str, Mapping[str, list[str]]] = DEFAULT_RULES,
    step: float = RULE_SCORE_STEP,
    minimum_score: float = RULE_SCORE_MIN,
    maximum_score: float = RULE_SCORE_MAX,
) -> pd.DataFrame:
    """
    Tính rule score dựa trên keyword của query và genres phim.

    Rule chỉ đóng vai trò boost hoặc penalty nhẹ,
    không thay thế semantic và Cross-Encoder ranking.
    """
    if candidates.empty:
        return candidates.copy()

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

    scores: list[float] = []

    for genres_value in candidates["genres"]:
        genres = {
            genre.strip()
            for genre in str(genres_value).split("|")
            if genre.strip()
        }

        score = 0.0

        score += step * len(
            genres.intersection(active_boost_genres)
        )

        score -= step * len(
            genres.intersection(active_penalty_genres)
        )

        scores.append(score)

    result = candidates.copy()

    result["rule_score"] = (
        pd.Series(scores, index=result.index)
        .clip(
            lower=minimum_score,
            upper=maximum_score,
        )
        .round(6)
    )

    return result


def compute_cross_encoder_score(
    candidates: pd.DataFrame,
    query: str,
    cross_encoder: CrossEncoder | None = None,
) -> pd.DataFrame:
    """
    Chấm Cross-Encoder score cho từng candidate.
    """
    if candidates.empty:
        return candidates.copy()

    _validate_candidate_columns(
        candidates,
        {"page_content"},
    )

    result = candidates.copy()

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

    result["cross_encoder_score"] = (
        pd.Series(scores, index=result.index)
        .astype("float64")
        .round(6)
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


def compute_final_score(
    candidates: pd.DataFrame,
    weight_cross_encoder: float = WEIGHT_CROSS_ENCODER,
    weight_semantic: float = WEIGHT_SEMANTIC,
    weight_popularity: float = WEIGHT_POPULARITY,
    weight_rule: float = WEIGHT_RULE,
) -> pd.DataFrame:
    """
    Tính final score và sắp xếp lại candidates.

    Công thức mặc định:

        0.50 * cross_encoder_score
        + 0.25 * semantic_similarity
        + 0.15 * popularity_score
        + 0.10 * rule_score
    """
    if candidates.empty:
        return candidates.copy()

    required_score_columns = {
        "cross_encoder_score",
        "semantic_similarity",
        "popularity_score",
        "rule_score",
    }

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

    result = candidates.copy()

    result["final_score"] = (
        weight_cross_encoder
        * result["cross_encoder_score"]
        + weight_semantic
        * result["semantic_similarity"]
        + weight_popularity
        * result["popularity_score"]
        + weight_rule
        * result["rule_score"]
    ).round(6)

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
) -> pd.DataFrame:
    """
    Chạy toàn bộ ranking trên candidate DataFrame đã retrieve.

    Pipeline:

        faiss_distance
        → semantic_similarity
        → popularity_score
        → rule_score
        → cross_encoder_score
        → final_score
    """
    if candidates.empty:
        return candidates.copy()

    _validate_candidate_columns(
        candidates,
        REQUIRED_CANDIDATE_COLUMNS,
    )

    result = compute_semantic_similarity(candidates)
    result = compute_popularity_score(result)
    result = compute_rule_score(
        result,
        query=query,
    )
    result = compute_cross_encoder_score(
        result,
        query=query,
        cross_encoder=cross_encoder,
    )
    result = compute_final_score(result)

    return result


def rank_movies(
    vector_store: FAISS,
    query: str,
    retrieval_k: int = RETRIEVAL_TOP_K,
    top_n: int = FINAL_RECOMMENDATION_TOP_K,
    cross_encoder: CrossEncoder | None = None,
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
    )

    return ranked_frame.head(top_n).reset_index(drop=True)