from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

from utils.bootstrap import ensure_project_root

ensure_project_root()

from data.src.config import (
    EVALUATION_CATEGORY_METRICS_PATH,
    EVALUATION_DIFFICULTY_METRICS_PATH,
    EVALUATION_OVERALL_METRICS_PATH,
    EVALUATION_PER_QUERY_METRICS_PATH,
    EVALUATION_SCORED_PREDICTIONS_PATH,
)

BASELINE_CONFIGURATION = "faiss_only"
DEFAULT_CHALLENGER_CONFIGURATION = "hybrid_with_ce"

OVERALL_REQUIRED_COLUMNS = (
    "configuration",
    "query_count",
    "precision_at_5_mean",
    "ndcg_at_5_mean",
    "mrr_at_5_mean",
    "mrr_strong_at_5_mean",
    "hit_rate_at_5_mean",
    "irrelevant_at_5_mean",
    "mean_relevance_at_5_mean",
)

CATEGORY_REQUIRED_COLUMNS = OVERALL_REQUIRED_COLUMNS + ("category",)
DIFFICULTY_REQUIRED_COLUMNS = OVERALL_REQUIRED_COLUMNS + ("difficulty",)

PER_QUERY_REQUIRED_COLUMNS = (
    "query_id",
    "query",
    "category",
    "difficulty",
    "configuration",
    "precision_at_5",
    "ndcg_at_5",
    "mrr_at_5",
    "mrr_strong_at_5",
    "hit_rate_at_5",
    "irrelevant_at_5",
    "mean_relevance_at_5",
)

SCORED_PREDICTIONS_REQUIRED_COLUMNS = (
    "query_id",
    "query",
    "category",
    "difficulty",
    "configuration",
    "rank",
    "movieId",
    "title",
    "genres",
    "faiss_distance",
    "semantic_similarity",
    "popularity_score",
    "rule_score",
    "cross_encoder_score",
    "evaluation_score",
    "relevance",
    "notes",
)

METRIC_COLUMNS = (
    "precision_at_5_mean",
    "ndcg_at_5_mean",
    "mrr_at_5_mean",
    "mrr_strong_at_5_mean",
    "hit_rate_at_5_mean",
    "irrelevant_at_5_mean",
    "mean_relevance_at_5_mean",
)

PER_QUERY_METRIC_COLUMNS = (
    "precision_at_5",
    "ndcg_at_5",
    "mrr_at_5",
    "mrr_strong_at_5",
    "hit_rate_at_5",
    "irrelevant_at_5",
    "mean_relevance_at_5",
)


class EvaluationDataError(ValueError):
    """Raised when an evaluation CSV is malformed or incomplete."""


@dataclass(frozen=True)
class EvaluationDataBundle:
    overall: pd.DataFrame
    category: pd.DataFrame
    difficulty: pd.DataFrame
    per_query: pd.DataFrame
    scored_predictions: pd.DataFrame | None = None


def _required_columns_as_list(required_columns: Iterable[str]) -> list[str]:
    return list(required_columns)


@st.cache_data(show_spinner=False)
def _read_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise EvaluationDataError(f"File dữ liệu rỗng: {path}")

    return df


def _validate_columns(
    df: pd.DataFrame,
    *,
    source_name: str,
    required_columns: Iterable[str],
) -> pd.DataFrame:
    required = _required_columns_as_list(required_columns)
    missing_columns = [column for column in required if column not in df.columns]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise EvaluationDataError(f"{source_name} thiếu cột bắt buộc: {missing_text}")

    return df.copy()


def load_evaluation_csv(
    path: Path,
    *,
    source_name: str,
    required_columns: Iterable[str],
) -> pd.DataFrame:
    df = _read_csv(str(path))

    return _validate_columns(
        df,
        source_name=source_name,
        required_columns=required_columns,
    )


def load_optional_evaluation_csv(
    path: Path,
    *,
    source_name: str,
    required_columns: Iterable[str],
) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        return load_evaluation_csv(
            path,
            source_name=source_name,
            required_columns=required_columns,
        )
    except (FileNotFoundError, EvaluationDataError):
        return None


def load_overall_metrics() -> pd.DataFrame:
    return load_evaluation_csv(
        EVALUATION_OVERALL_METRICS_PATH,
        source_name="overall_metrics.csv",
        required_columns=OVERALL_REQUIRED_COLUMNS,
    )


def load_category_metrics() -> pd.DataFrame:
    return load_evaluation_csv(
        EVALUATION_CATEGORY_METRICS_PATH,
        source_name="category_metrics.csv",
        required_columns=CATEGORY_REQUIRED_COLUMNS,
    )


def load_difficulty_metrics() -> pd.DataFrame:
    return load_evaluation_csv(
        EVALUATION_DIFFICULTY_METRICS_PATH,
        source_name="difficulty_metrics.csv",
        required_columns=DIFFICULTY_REQUIRED_COLUMNS,
    )


def load_per_query_metrics() -> pd.DataFrame:
    return load_evaluation_csv(
        EVALUATION_PER_QUERY_METRICS_PATH,
        source_name="per_query_metrics.csv",
        required_columns=PER_QUERY_REQUIRED_COLUMNS,
    )


def load_scored_predictions() -> pd.DataFrame | None:
    return load_optional_evaluation_csv(
        EVALUATION_SCORED_PREDICTIONS_PATH,
        source_name="scored_predictions.csv",
        required_columns=SCORED_PREDICTIONS_REQUIRED_COLUMNS,
    )


def format_configuration(name: str) -> str:
    mapping = {
        "hybrid_with_ce": "Hybrid + Cross Encoder",
        "hybrid_no_ce": "Hybrid",
        "cross_encoder_only": "Cross Encoder",
        "faiss_only": "FAISS",
    }

    return mapping.get(name, name)


def get_best_row(
    df: pd.DataFrame,
    metric: str = "ndcg_at_5_mean",
) -> pd.Series:
    return df.sort_values(metric, ascending=False).iloc[0]


def get_configuration_row(
    df: pd.DataFrame,
    configuration: str,
) -> pd.Series:
    matches = df[df["configuration"] == configuration]

    if matches.empty:
        raise EvaluationDataError(
            f"Không tìm thấy configuration '{configuration}' trong bảng dữ liệu."
        )

    return matches.iloc[0]


def compare_to_baseline(
    df: pd.DataFrame,
    *,
    metric_columns: Iterable[str],
    baseline_configuration: str = BASELINE_CONFIGURATION,
    challenger_configuration: str = DEFAULT_CHALLENGER_CONFIGURATION,
) -> dict[str, float | str]:
    baseline_row = get_configuration_row(df, baseline_configuration)
    challenger_row = get_configuration_row(df, challenger_configuration)

    result: dict[str, float | str] = {
        "baseline_configuration": baseline_configuration,
        "challenger_configuration": challenger_configuration,
    }

    for column in metric_columns:
        baseline_value = float(baseline_row[column])
        challenger_value = float(challenger_row[column])
        delta = challenger_value - baseline_value
        pct_change = None if baseline_value == 0 else delta / baseline_value * 100

        result[f"baseline_{column}"] = baseline_value
        result[f"challenger_{column}"] = challenger_value
        result[f"delta_{column}"] = delta
        result[f"pct_change_{column}"] = pct_change if pct_change is not None else 0.0

    return result


def build_overall_headlines(
    overall_df: pd.DataFrame,
) -> dict[str, float | str]:
    best_row = get_best_row(overall_df)
    baseline_row = get_configuration_row(overall_df, BASELINE_CONFIGURATION)

    return {
        "best_configuration": best_row["configuration"],
        "best_precision_at_5_mean": float(best_row["precision_at_5_mean"]),
        "best_ndcg_at_5_mean": float(best_row["ndcg_at_5_mean"]),
        "best_mrr_at_5_mean": float(best_row["mrr_at_5_mean"]),
        "best_mean_relevance_at_5_mean": float(best_row["mean_relevance_at_5_mean"]),
        "faiss_precision_at_5_mean": float(baseline_row["precision_at_5_mean"]),
        "faiss_ndcg_at_5_mean": float(baseline_row["ndcg_at_5_mean"]),
        "faiss_mrr_at_5_mean": float(baseline_row["mrr_at_5_mean"]),
        "faiss_mean_relevance_at_5_mean": float(
            baseline_row["mean_relevance_at_5_mean"]
        ),
    }


def build_metric_delta_table(
    df: pd.DataFrame,
    *,
    metric_columns: Iterable[str],
    baseline_configuration: str = BASELINE_CONFIGURATION,
    challenger_configuration: str = DEFAULT_CHALLENGER_CONFIGURATION,
) -> pd.DataFrame:
    comparison = compare_to_baseline(
        df,
        metric_columns=metric_columns,
        baseline_configuration=baseline_configuration,
        challenger_configuration=challenger_configuration,
    )

    rows = []

    for column in metric_columns:
        rows.append(
            {
                "metric": column,
                "baseline": comparison[f"baseline_{column}"],
                "challenger": comparison[f"challenger_{column}"],
                "delta": comparison[f"delta_{column}"],
                "pct_change": comparison[f"pct_change_{column}"],
            }
        )

    return pd.DataFrame(rows)


def build_per_query_comparison_table(
    per_query_df: pd.DataFrame,
    *,
    metric: str = "ndcg_at_5",
    baseline_configuration: str = BASELINE_CONFIGURATION,
    challenger_configuration: str = DEFAULT_CHALLENGER_CONFIGURATION,
) -> pd.DataFrame:
    pivot = per_query_df.pivot_table(
        index=["query_id", "query", "category", "difficulty"],
        columns="configuration",
        values=metric,
        aggfunc="first",
    ).reset_index()

    if (
        baseline_configuration not in pivot.columns
        or challenger_configuration not in pivot.columns
    ):
        raise EvaluationDataError(
            "Bảng per_query_metrics không có đủ configuration để so sánh case study."
        )

    comparison = pivot.copy()
    comparison[f"delta_{metric}"] = (
        comparison[challenger_configuration] - comparison[baseline_configuration]
    )
    comparison[f"pct_change_{metric}"] = comparison.apply(
        lambda row: (
            0.0
            if row[baseline_configuration] == 0
            else row[f"delta_{metric}"] / row[baseline_configuration] * 100
        ),
        axis=1,
    )

    return comparison.sort_values(f"delta_{metric}", ascending=False)


def build_case_study_candidates(
    per_query_df: pd.DataFrame,
    *,
    metric: str = "ndcg_at_5",
    top_n: int = 3,
    baseline_configuration: str = BASELINE_CONFIGURATION,
    challenger_configuration: str = DEFAULT_CHALLENGER_CONFIGURATION,
) -> pd.DataFrame:
    comparison = build_per_query_comparison_table(
        per_query_df,
        metric=metric,
        baseline_configuration=baseline_configuration,
        challenger_configuration=challenger_configuration,
    )

    return comparison.nlargest(top_n, f"delta_{metric}")


def get_case_study_queries(scored_predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        scored_predictions[["query_id", "query", "category", "difficulty"]]
        .drop_duplicates()
        .sort_values(["category", "difficulty", "query_id"])
        .reset_index(drop=True)
    )


def get_case_study_query_metadata(
    scored_predictions: pd.DataFrame,
    query_id: str,
) -> dict[str, str]:
    matches = scored_predictions[scored_predictions["query_id"] == query_id]

    if matches.empty:
        raise EvaluationDataError(
            f"Không tìm thấy query_id '{query_id}' trong scored_predictions.csv"
        )

    row = matches.iloc[0]

    return {
        "query_id": str(row["query_id"]),
        "query": str(row["query"]),
        "category": str(row["category"]),
        "difficulty": str(row["difficulty"]),
    }


def build_case_study_rankings(
    scored_predictions: pd.DataFrame,
    query_id: str,
    *,
    configurations: Iterable[str] = (
        BASELINE_CONFIGURATION,
        DEFAULT_CHALLENGER_CONFIGURATION,
    ),
    top_n: int = 5,
) -> dict[str, pd.DataFrame]:
    query_rows = scored_predictions[scored_predictions["query_id"] == query_id]

    if query_rows.empty:
        raise EvaluationDataError(
            f"Không tìm thấy dữ liệu case study cho query_id '{query_id}'."
        )

    rankings: dict[str, pd.DataFrame] = {}
    columns = [
        "rank",
        "movieId",
        "title",
        "genres",
        "faiss_distance",
        "semantic_similarity",
        "popularity_score",
        "rule_score",
        "cross_encoder_score",
        "evaluation_score",
        "relevance",
        "notes",
    ]

    for configuration in configurations:
        config_rows = (
            query_rows[query_rows["configuration"] == configuration]
            .sort_values("rank")
            .head(top_n)
            .copy()
        )

        if config_rows.empty:
            continue

        rankings[configuration] = config_rows[columns].reset_index(drop=True)

    if not rankings:
        raise EvaluationDataError(
            f"Query '{query_id}' không có configuration hợp lệ để hiển thị."
        )

    return rankings


def build_evaluation_bundle() -> EvaluationDataBundle:
    scored_predictions = load_scored_predictions()

    return EvaluationDataBundle(
        overall=load_overall_metrics(),
        category=load_category_metrics(),
        difficulty=load_difficulty_metrics(),
        per_query=load_per_query_metrics(),
        scored_predictions=scored_predictions,
    )
