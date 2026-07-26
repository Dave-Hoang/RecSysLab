from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_TOP_K = 5
RELEVANT_THRESHOLD = 1
STRONG_RELEVANCE_VALUE = 2

REQUIRED_COLUMNS = {
    "query_id",
    "query",
    "category",
    "difficulty",
    "configuration",
    "rank",
    "movieId",
    "relevance",
}

METRIC_COLUMNS = [
    "precision_at_5",
    "ndcg_at_5",
    "mrr_at_5",
    "mrr_strong_at_5",
    "hit_rate_at_5",
    "irrelevant_at_5",
    "mean_relevance_at_5",
]


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Các DataFrame kết quả được tạo trong giai đoạn evaluation."""

    per_query: pd.DataFrame
    overall: pd.DataFrame
    by_category: pd.DataFrame
    by_difficulty: pd.DataFrame


def _validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataframe_name: str,
) -> None:
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{dataframe_name} thiếu các cột bắt buộc: "
            f"{sorted(missing_columns)}"
        )


def load_scored_predictions(path: Path) -> pd.DataFrame:
    """
    Đọc và chuẩn hóa scored_predictions.csv.

    Không tự điền relevance bị thiếu. Nếu dữ liệu không hợp lệ,
    hàm sẽ dừng để tránh tạo metric sai.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy scored predictions file: {path}"
        )

    dataframe = pd.read_csv(
        path,
        dtype={
            "query_id": "string",
            "query": "string",
            "category": "string",
            "difficulty": "string",
            "configuration": "string",
            "title": "string",
            "genres": "string",
            "notes": "string",
        },
    )

    _validate_required_columns(
        dataframe=dataframe,
        required_columns=REQUIRED_COLUMNS,
        dataframe_name="scored_predictions.csv",
    )

    if dataframe.empty:
        raise ValueError("scored_predictions.csv không có dữ liệu.")

    for column in [
        "query_id",
        "query",
        "category",
        "difficulty",
        "configuration",
    ]:
        dataframe[column] = dataframe[column].astype("string").str.strip()

    dataframe["movieId"] = pd.to_numeric(
        dataframe["movieId"],
        errors="raise",
    ).astype("int64")

    dataframe["rank"] = pd.to_numeric(
        dataframe["rank"],
        errors="raise",
    ).astype("int64")

    dataframe["relevance"] = pd.to_numeric(
        dataframe["relevance"],
        errors="coerce",
    )

    if dataframe["relevance"].isna().any():
        invalid_rows = dataframe.loc[
            dataframe["relevance"].isna(),
            ["query_id", "configuration", "movieId", "rank"],
        ]

        raise ValueError(
            "Có prediction thiếu relevance:\n"
            f"{invalid_rows.to_string(index=False)}"
        )

    dataframe["relevance"] = dataframe["relevance"].astype("int8")

    invalid_relevance = dataframe.loc[
        ~dataframe["relevance"].isin([0, 1, 2]),
        ["query_id", "configuration", "movieId", "relevance"],
    ]

    if not invalid_relevance.empty:
        raise ValueError(
            "Relevance chỉ được nhận giá trị 0, 1 hoặc 2:\n"
            f"{invalid_relevance.to_string(index=False)}"
        )

    return dataframe


def validate_scored_predictions(
    dataframe: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
    expected_query_count: int | None = 30,
    expected_configuration_count: int | None = 4,
) -> None:
    """Kiểm tra cấu trúc benchmark trước khi tính metric."""

    if top_k <= 0:
        raise ValueError("top_k phải lớn hơn 0.")

    if expected_query_count is not None:
        actual_query_count = dataframe["query_id"].nunique()

        if actual_query_count != expected_query_count:
            raise ValueError(
                f"Expected {expected_query_count} query nhưng "
                f"tìm thấy {actual_query_count}."
            )

    if expected_configuration_count is not None:
        actual_configuration_count = (
            dataframe["configuration"].nunique()
        )

        if actual_configuration_count != expected_configuration_count:
            raise ValueError(
                f"Expected {expected_configuration_count} configuration "
                f"nhưng tìm thấy {actual_configuration_count}."
            )

    invalid_ranks = dataframe.loc[
        ~dataframe["rank"].between(1, top_k),
        [
            "query_id",
            "configuration",
            "movieId",
            "rank",
        ],
    ]

    if not invalid_ranks.empty:
        raise ValueError(
            f"Có rank nằm ngoài khoảng 1–{top_k}:\n"
            f"{invalid_ranks.to_string(index=False)}"
        )

    group_sizes = (
        dataframe
        .groupby(
            ["query_id", "configuration"],
            observed=True,
        )
        .size()
    )

    invalid_groups = group_sizes.loc[group_sizes != top_k]

    if not invalid_groups.empty:
        raise ValueError(
            "Một số query × configuration không có đúng "
            f"{top_k} kết quả:\n"
            f"{invalid_groups.to_string()}"
        )

    duplicated_ranks = dataframe.duplicated(
        subset=["query_id", "configuration", "rank"],
        keep=False,
    )

    if duplicated_ranks.any():
        duplicate_rows = dataframe.loc[
            duplicated_ranks,
            [
                "query_id",
                "configuration",
                "rank",
                "movieId",
            ],
        ]

        raise ValueError(
            "Có rank bị trùng trong cùng query và configuration:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    duplicated_movies = dataframe.duplicated(
        subset=["query_id", "configuration", "movieId"],
        keep=False,
    )

    if duplicated_movies.any():
        duplicate_rows = dataframe.loc[
            duplicated_movies,
            [
                "query_id",
                "configuration",
                "rank",
                "movieId",
            ],
        ]

        raise ValueError(
            "Một phim xuất hiện nhiều lần trong cùng query và "
            "configuration:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    # Một query-movie phải luôn có cùng relevance ở mọi configuration.
    relevance_conflicts = (
        dataframe
        .groupby(
            ["query_id", "movieId"],
            observed=True,
        )["relevance"]
        .nunique()
    )

    relevance_conflicts = relevance_conflicts.loc[
        relevance_conflicts > 1
    ]

    if not relevance_conflicts.empty:
        conflict_keys = relevance_conflicts.reset_index()[
            ["query_id", "movieId"]
        ]

        conflict_rows = dataframe.merge(
            conflict_keys,
            on=["query_id", "movieId"],
            how="inner",
        )[
            [
                "query_id",
                "movieId",
                "configuration",
                "relevance",
            ]
        ]

        raise ValueError(
            "Một số query_id + movieId có relevance không nhất quán:\n"
            f"{conflict_rows.to_string(index=False)}"
        )


def dcg_at_k(relevances: Iterable[int | float], k: int) -> float:
    """
    Tính Discounted Cumulative Gain với graded relevance.

    gain = 2^relevance - 1
    discount = log2(rank + 1)
    """

    values = np.asarray(list(relevances), dtype=float)[:k]

    if values.size == 0:
        return 0.0

    discounts = np.log2(
        np.arange(2, values.size + 2, dtype=float)
    )
    gains = np.power(2.0, values) - 1.0

    return float(np.sum(gains / discounts))


def _build_query_label_pool(
    dataframe: pd.DataFrame,
) -> dict[str, list[int]]:
    """
    Tạo pool ground truth duy nhất cho từng query.

    Pool được lấy từ union candidate đã gán nhãn, không lấy riêng
    Top 5 của từng configuration. Nhờ vậy IDCG của cùng một query
    giống nhau ở cả bốn cấu hình.
    """

    pool = (
        dataframe[
            ["query_id", "movieId", "relevance"]
        ]
        .drop_duplicates(
            subset=["query_id", "movieId"],
            keep="first",
        )
    )

    return {
        str(query_id): (
            group["relevance"]
            .sort_values(ascending=False)
            .astype(int)
            .tolist()
        )
        for query_id, group in pool.groupby(
            "query_id",
            sort=False,
            observed=True,
        )
    }


def ndcg_at_k(
    ranked_relevances: Iterable[int | float],
    ideal_relevances: Iterable[int | float],
    k: int,
) -> float:
    """Tính NDCG@K bằng cùng một ideal pool cho mỗi query."""

    dcg = dcg_at_k(ranked_relevances, k=k)
    idcg = dcg_at_k(ideal_relevances, k=k)

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)


def reciprocal_rank_at_k(
    relevances: Iterable[int | float],
    k: int,
    threshold: int = RELEVANT_THRESHOLD,
) -> float:
    """
    Reciprocal rank của kết quả relevant đầu tiên.

    Với threshold=1, relevance 1 hoặc 2 được coi là relevant.
    Với threshold=2, chỉ relevance 2 được coi là strong relevant.
    """

    for position, relevance in enumerate(
        list(relevances)[:k],
        start=1,
    ):
        if relevance >= threshold:
            return 1.0 / position

    return 0.0


def compute_per_query_metrics(
    dataframe: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """
    Tính metric cho từng query_id × configuration.

    Output dự kiến:
        số query × số configuration
        = 30 × 4
        = 120 dòng
    """

    query_label_pool = _build_query_label_pool(dataframe)
    records: list[dict[str, object]] = []

    grouped = dataframe.groupby(
        ["query_id", "configuration"],
        sort=False,
        observed=True,
    )

    for (query_id, configuration), group in grouped:
        ranked = group.sort_values(
            by="rank",
            kind="stable",
        ).head(top_k)

        relevances = ranked["relevance"].astype(int).tolist()
        binary_relevant = [
            int(value >= RELEVANT_THRESHOLD)
            for value in relevances
        ]

        first_row = ranked.iloc[0]
        ideal_relevances = query_label_pool[str(query_id)]

        records.append(
            {
                "query_id": str(query_id),
                "query": first_row["query"],
                "category": first_row["category"],
                "difficulty": first_row["difficulty"],
                "configuration": str(configuration),
                "precision_at_5": float(
                    np.mean(binary_relevant)
                ),
                "ndcg_at_5": ndcg_at_k(
                    ranked_relevances=relevances,
                    ideal_relevances=ideal_relevances,
                    k=top_k,
                ),
                "mrr_at_5": reciprocal_rank_at_k(
                    relevances,
                    k=top_k,
                    threshold=RELEVANT_THRESHOLD,
                ),
                "mrr_strong_at_5": reciprocal_rank_at_k(
                    relevances,
                    k=top_k,
                    threshold=STRONG_RELEVANCE_VALUE,
                ),
                "hit_rate_at_5": float(
                    any(binary_relevant)
                ),
                "irrelevant_at_5": int(
                    sum(value == 0 for value in relevances)
                ),
                "mean_relevance_at_5": float(
                    np.mean(relevances)
                ),
            }
        )

    per_query = pd.DataFrame.from_records(records)

    return per_query.sort_values(
        by=["query_id", "configuration"],
        kind="stable",
    ).reset_index(drop=True)


def _flatten_aggregated_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    flattened = dataframe.copy()

    flattened.columns = [
        "_".join(
            str(part)
            for part in column
            if str(part)
        )
        if isinstance(column, tuple)
        else str(column)
        for column in flattened.columns
    ]

    return flattened.reset_index()


def aggregate_metrics(
    per_query_metrics: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Tổng hợp mean, std và median cho các metric."""

    aggregation = {
        metric: ["mean", "std", "median"]
        for metric in METRIC_COLUMNS
    }

    aggregated = (
        per_query_metrics
        .groupby(
            group_columns,
            observed=True,
            sort=False,
        )
        .agg(aggregation)
    )

    aggregated = _flatten_aggregated_columns(aggregated)

    query_counts = (
        per_query_metrics
        .groupby(
            group_columns,
            observed=True,
            sort=False,
        )["query_id"]
        .nunique()
        .rename("query_count")
        .reset_index()
    )

    aggregated = aggregated.merge(
        query_counts,
        on=group_columns,
        how="left",
        validate="one_to_one",
    )

    ordered_columns = (
        group_columns
        + ["query_count"]
        + [
            column
            for metric in METRIC_COLUMNS
            for column in [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_median",
            ]
        ]
    )

    return aggregated[ordered_columns]


def compute_all_metrics(
    dataframe: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
    expected_query_count: int | None = 30,
    expected_configuration_count: int | None = 4,
) -> EvaluationArtifacts:
    """Chạy toàn bộ giai đoạn tính metric."""

    validate_scored_predictions(
        dataframe=dataframe,
        top_k=top_k,
        expected_query_count=expected_query_count,
        expected_configuration_count=(
            expected_configuration_count
        ),
    )

    per_query = compute_per_query_metrics(
        dataframe=dataframe,
        top_k=top_k,
    )

    expected_rows = (
        dataframe["query_id"].nunique()
        * dataframe["configuration"].nunique()
    )

    if len(per_query) != expected_rows:
        raise RuntimeError(
            "Số dòng per-query metrics không đúng. "
            f"Expected {expected_rows}, actual {len(per_query)}."
        )

    overall = aggregate_metrics(
        per_query_metrics=per_query,
        group_columns=["configuration"],
    ).sort_values(
        by="ndcg_at_5_mean",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)

    by_category = aggregate_metrics(
        per_query_metrics=per_query,
        group_columns=["configuration", "category"],
    ).sort_values(
        by=["category", "ndcg_at_5_mean"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)

    by_difficulty = aggregate_metrics(
        per_query_metrics=per_query,
        group_columns=["configuration", "difficulty"],
    ).sort_values(
        by=["difficulty", "ndcg_at_5_mean"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)

    return EvaluationArtifacts(
        per_query=per_query,
        overall=overall,
        by_category=by_category,
        by_difficulty=by_difficulty,
    )


def save_evaluation_artifacts(
    artifacts: EvaluationArtifacts,
    output_dir: Path,
) -> dict[str, Path]:
    """Lưu bốn file CSV bằng UTF-8 BOM để mở tốt trong Excel."""

    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "per_query": output_dir / "per_query_metrics.csv",
        "overall": output_dir / "overall_metrics.csv",
        "category": output_dir / "category_metrics.csv",
        "difficulty": output_dir / "difficulty_metrics.csv",
    }

    artifacts.per_query.to_csv(
        output_paths["per_query"],
        index=False,
        encoding="utf-8-sig",
    )
    artifacts.overall.to_csv(
        output_paths["overall"],
        index=False,
        encoding="utf-8-sig",
    )
    artifacts.by_category.to_csv(
        output_paths["category"],
        index=False,
        encoding="utf-8-sig",
    )
    artifacts.by_difficulty.to_csv(
        output_paths["difficulty"],
        index=False,
        encoding="utf-8-sig",
    )

    return output_paths