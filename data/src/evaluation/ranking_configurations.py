from __future__ import annotations

import pandas as pd

from src.evaluation.constants import (
    HYBRID_NO_CE_WEIGHTS,
    HYBRID_WITH_CE_WEIGHTS,
    RankingConfiguration,
)
from src.ranking.hybrid_ranker import (
    compute_cross_encoder_score,
    compute_popularity_score,
    compute_rule_score,
    compute_semantic_similarity,
)


REQUIRED_CANDIDATE_COLUMNS = {
    "rank",
    "movieId",
    "title",
    "genres",
    "rating_mean",
    "rating_count",
    "page_content",
}

SCORE_COLUMNS = {
    "semantic_similarity",
    "popularity_score",
    "rule_score",
    "cross_encoder_score",
}


def _validate_candidates(
    candidates: pd.DataFrame,
) -> None:
    """Kiểm tra candidate pool trước khi ranking."""

    if not isinstance(candidates, pd.DataFrame):
        raise TypeError(
            "candidates phải là một pandas DataFrame."
        )

    if candidates.empty:
        raise ValueError(
            "Candidate DataFrame không được rỗng."
        )

    missing_columns = (
        REQUIRED_CANDIDATE_COLUMNS
        - set(candidates.columns)
    )

    if missing_columns:
        raise ValueError(
            "Candidate DataFrame thiếu cột: "
            f"{sorted(missing_columns)}"
        )

    if candidates["movieId"].isna().any():
        raise ValueError(
            "Cột movieId chứa giá trị trống."
        )

    if candidates["rank"].isna().any():
        raise ValueError(
            "Cột rank chứa giá trị trống."
        )


def _validate_top_k(
    top_k: int,
) -> None:
    """Kiểm tra giá trị Top-K."""

    if not isinstance(top_k, int):
        raise TypeError(
            "top_k phải là số nguyên."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k phải lớn hơn 0."
        )


def _prepare_candidates(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Chuẩn hóa candidate pool.

    Cột rank từ retriever được lưu thành retrieval_rank để
    phân biệt với thứ hạng kết quả sau reranking.
    """

    frame = candidates.copy(deep=True)

    frame["movieId"] = pd.to_numeric(
        frame["movieId"],
        errors="raise",
    ).astype(int)

    frame["retrieval_rank"] = pd.to_numeric(
        frame["rank"],
        errors="coerce",
    )

    if frame["retrieval_rank"].isna().any():
        raise ValueError(
            "Cột rank chứa giá trị không hợp lệ."
        )

    frame["retrieval_rank"] = (
        frame["retrieval_rank"].astype(int)
    )

    if (frame["retrieval_rank"] <= 0).any():
        raise ValueError(
            "retrieval_rank phải bắt đầu từ 1."
        )

    if frame["movieId"].duplicated().any():
        frame = (
            frame.sort_values(
                by=["retrieval_rank", "movieId"],
                ascending=[True, True],
            )
            .drop_duplicates(
                subset=["movieId"],
                keep="first",
            )
            .reset_index(drop=True)
        )

    return frame


def _add_rank_based_semantic_score(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tạo semantic score từ retrieval rank.

    Chỉ dùng khi retriever không cung cấp FAISS distance hoặc
    similarity score gốc.

    Đây là normalized retrieval score, không phải cosine
    similarity thật.
    """

    frame = candidates.copy()

    candidate_count = len(frame)

    if candidate_count == 1:
        frame["semantic_similarity"] = 1.0
    else:
        minimum_rank = frame["retrieval_rank"].min()
        maximum_rank = frame["retrieval_rank"].max()

        rank_range = maximum_rank - minimum_rank

        if rank_range == 0:
            frame["semantic_similarity"] = 1.0
        else:
            frame["semantic_similarity"] = (
                1.0
                - (
                    frame["retrieval_rank"]
                    - minimum_rank
                )
                / rank_range
            )

    frame["semantic_similarity"] = (
        pd.to_numeric(
            frame["semantic_similarity"],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
        .astype(float)
    )

    return frame


def _add_semantic_score(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tính semantic score.

    Nếu có faiss_distance hợp lệ thì dùng hàm production
    compute_semantic_similarity(). Nếu không, dùng retrieval
    rank làm normalized retrieval score.
    """

    frame = candidates.copy()

    has_faiss_distance = (
        "faiss_distance" in frame.columns
        and frame["faiss_distance"].notna().any()
    )

    if has_faiss_distance:
        frame = compute_semantic_similarity(frame)
    else:
        frame["faiss_distance"] = pd.NA
        frame = _add_rank_based_semantic_score(frame)

    if "semantic_similarity" not in frame.columns:
        raise ValueError(
            "Không tạo được cột semantic_similarity."
        )

    frame["semantic_similarity"] = (
        pd.to_numeric(
            frame["semantic_similarity"],
            errors="coerce",
        )
        .fillna(0.0)
        .astype(float)
    )

    return frame


def _normalize_score_column(
    frame: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Chuyển một score column thành kiểu float an toàn."""

    if column not in frame.columns:
        raise ValueError(
            f"Không tìm thấy cột score: {column}"
        )

    frame[column] = (
        pd.to_numeric(
            frame[column],
            errors="coerce",
        )
        .fillna(0.0)
        .astype(float)
    )

    return frame


def _add_popularity_score(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    frame = compute_popularity_score(
        candidates.copy()
    )

    return _normalize_score_column(
        frame=frame,
        column="popularity_score",
    )


def _add_rule_score(
    query: str,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    frame = compute_rule_score(
        query=query,
        candidates=candidates.copy(),
    )

    return _normalize_score_column(
        frame=frame,
        column="rule_score",
    )


def _add_cross_encoder_score(
    query: str,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    frame = compute_cross_encoder_score(
        query=query,
        candidates=candidates.copy(),
    )

    return _normalize_score_column(
        frame=frame,
        column="cross_encoder_score",
    )


def _prepare_base_scores(
    query: str,
    candidates: pd.DataFrame,
    include_cross_encoder: bool,
) -> pd.DataFrame:
    """Tính các thành phần score cần cho hybrid ranking."""

    frame = _prepare_candidates(candidates)
    frame = _add_semantic_score(frame)
    frame = _add_popularity_score(frame)
    frame = _add_rule_score(
        query=query,
        candidates=frame,
    )

    if include_cross_encoder:
        frame = _add_cross_encoder_score(
            query=query,
            candidates=frame,
        )
    else:
        frame["cross_encoder_score"] = 0.0

    return frame


def _finalize_ranking(
    frame: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Sắp xếp và trả về Top-K kết quả."""

    if "evaluation_score" not in frame.columns:
        raise ValueError(
            "Không tìm thấy cột evaluation_score."
        )

    frame["evaluation_score"] = (
        pd.to_numeric(
            frame["evaluation_score"],
            errors="coerce",
        )
        .fillna(0.0)
        .astype(float)
    )

    result = (
        frame.sort_values(
            by=[
                "evaluation_score",
                "retrieval_rank",
                "movieId",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    result["result_rank"] = (
        result.index + 1
    )

    return result


def rank_faiss_only(
    candidates: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """
    Baseline FAISS.

    Giữ nguyên thứ tự candidate pool được trả về từ FAISS.
    """

    _validate_candidates(candidates)
    _validate_top_k(top_k)

    frame = _prepare_candidates(candidates)
    frame = _add_semantic_score(frame)

    frame["popularity_score"] = 0.0
    frame["rule_score"] = 0.0
    frame["cross_encoder_score"] = 0.0
    frame["evaluation_score"] = (
        frame["semantic_similarity"]
    )

    result = (
        frame.sort_values(
            by=["retrieval_rank", "movieId"],
            ascending=[True, True],
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    result["result_rank"] = (
        result.index + 1
    )

    return result


def rank_hybrid_no_ce(
    query: str,
    candidates: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Hybrid ranking không sử dụng Cross-Encoder."""

    _validate_candidates(candidates)
    _validate_top_k(top_k)

    frame = _prepare_base_scores(
        query=query,
        candidates=candidates,
        include_cross_encoder=False,
    )

    weights = HYBRID_NO_CE_WEIGHTS

    frame["evaluation_score"] = (
        weights["semantic"]
        * frame["semantic_similarity"]
        + weights["popularity"]
        * frame["popularity_score"]
        + weights["rule"]
        * frame["rule_score"]
    )

    return _finalize_ranking(
        frame=frame,
        top_k=top_k,
    )


def rank_cross_encoder_only(
    query: str,
    candidates: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Reranking chỉ dựa trên Cross-Encoder."""

    _validate_candidates(candidates)
    _validate_top_k(top_k)

    frame = _prepare_candidates(candidates)
    frame = _add_semantic_score(frame)

    frame["popularity_score"] = 0.0
    frame["rule_score"] = 0.0

    frame = _add_cross_encoder_score(
        query=query,
        candidates=frame,
    )

    frame["evaluation_score"] = (
        frame["cross_encoder_score"]
    )

    return _finalize_ranking(
        frame=frame,
        top_k=top_k,
    )


def rank_hybrid_with_ce(
    query: str,
    candidates: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Hybrid ranking đầy đủ có Cross-Encoder."""

    _validate_candidates(candidates)
    _validate_top_k(top_k)

    frame = _prepare_base_scores(
        query=query,
        candidates=candidates,
        include_cross_encoder=True,
    )

    weights = HYBRID_WITH_CE_WEIGHTS

    frame["evaluation_score"] = (
        weights["cross_encoder"]
        * frame["cross_encoder_score"]
        + weights["semantic"]
        * frame["semantic_similarity"]
        + weights["popularity"]
        * frame["popularity_score"]
        + weights["rule"]
        * frame["rule_score"]
    )

    return _finalize_ranking(
        frame=frame,
        top_k=top_k,
    )


def rank_with_configuration(
    query: str,
    candidates: pd.DataFrame,
    configuration: RankingConfiguration,
    top_k: int,
) -> pd.DataFrame:
    """Điều phối ranking theo ablation configuration."""

    if isinstance(configuration, str):
        try:
            configuration = RankingConfiguration(
                configuration
            )
        except ValueError as error:
            raise ValueError(
                "Ranking configuration không hợp lệ: "
                f"{configuration}"
            ) from error

    if configuration == RankingConfiguration.FAISS_ONLY:
        return rank_faiss_only(
            candidates=candidates,
            top_k=top_k,
        )

    if configuration == RankingConfiguration.HYBRID_NO_CE:
        return rank_hybrid_no_ce(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

    if (
        configuration
        == RankingConfiguration.CROSS_ENCODER_ONLY
    ):
        return rank_cross_encoder_only(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

    if (
        configuration
        == RankingConfiguration.HYBRID_WITH_CE
    ):
        return rank_hybrid_with_ce(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

    raise ValueError(
        "Ranking configuration không được hỗ trợ: "
        f"{configuration}"
    )