from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import (
    EVALUATION_METRICS_DIR,
    EVALUATION_SCORED_PREDICTIONS_PATH,
)
from src.evaluation.metrics import (
    compute_all_metrics,
    load_scored_predictions,
    save_evaluation_artifacts,
)


EVALUATION_TOP_K = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tính Precision@5, NDCG@5, MRR@5, Hit Rate@5 "
            "và các bảng tổng hợp cho bốn cấu hình."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=EVALUATION_SCORED_PREDICTIONS_PATH,
        help=(
            "Đường dẫn scored_predictions.csv. "
            f"Mặc định: {EVALUATION_SCORED_PREDICTIONS_PATH}"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVALUATION_METRICS_DIR,
        help=(
            "Thư mục lưu các file metric. "
            f"Mặc định: {EVALUATION_METRICS_DIR}"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        choices=[EVALUATION_TOP_K],
        default=EVALUATION_TOP_K,
        help=(
            "K dùng để đánh giá. Benchmark hiện tại "
            f"được cố định ở K={EVALUATION_TOP_K}."
        ),
    )

    parser.add_argument(
        "--expected-query-count",
        type=int,
        default=30,
        help="Số query kỳ vọng. Mặc định: 30.",
    )

    parser.add_argument(
        "--expected-configuration-count",
        type=int,
        default=4,
        help="Số configuration kỳ vọng. Mặc định: 4.",
    )

    return parser


def _print_overall_summary(
    overall: pd.DataFrame,
) -> None:
    if overall.empty:
        raise ValueError(
            "overall_metrics không có dữ liệu để hiển thị."
        )

    columns = [
        "configuration",
        "precision_at_5_mean",
        "ndcg_at_5_mean",
        "mrr_at_5_mean",
        "mrr_strong_at_5_mean",
        "hit_rate_at_5_mean",
        "irrelevant_at_5_mean",
        "mean_relevance_at_5_mean",
    ]

    missing_columns = set(columns).difference(
        overall.columns
    )

    if missing_columns:
        raise ValueError(
            "overall_metrics thiếu các cột bắt buộc: "
            f"{sorted(missing_columns)}"
        )

    summary = overall[columns].copy()

    metric_columns = [
        column
        for column in columns
        if column != "configuration"
    ]

    summary[metric_columns] = (
        summary[metric_columns].round(4)
    )

    print("\nOVERALL METRICS")
    print("-" * 110)
    print(summary.to_string(index=False))


def _validate_arguments(
    args: argparse.Namespace,
) -> None:
    if args.expected_query_count <= 0:
        raise ValueError(
            "--expected-query-count phải lớn hơn 0."
        )

    if args.expected_configuration_count <= 0:
        raise ValueError(
            "--expected-configuration-count phải lớn hơn 0."
        )


def main() -> None:
    args = build_parser().parse_args()
    _validate_arguments(args)

    print("=" * 72)
    print(
        "COMPUTE MOVIE RECOMMENDATION "
        "EVALUATION METRICS"
    )
    print("=" * 72)

    print(f"Input      : {args.input}")
    print(f"Output dir : {args.output_dir}")
    print(f"Top K      : {args.top_k}")

    scored_predictions = load_scored_predictions(
        args.input
    )

    query_count = (
        scored_predictions["query_id"].nunique()
    )

    configuration_count = (
        scored_predictions["configuration"].nunique()
    )

    print(
        "\nDữ liệu đầu vào:"
        f"\n- Rows          : "
        f"{len(scored_predictions):,}"
        f"\n- Queries       : {query_count}"
        f"\n- Configurations: {configuration_count}"
    )

    artifacts = compute_all_metrics(
        dataframe=scored_predictions,
        top_k=args.top_k,
        expected_query_count=(
            args.expected_query_count
        ),
        expected_configuration_count=(
            args.expected_configuration_count
        ),
    )

    expected_per_query_rows = (
        args.expected_query_count
        * args.expected_configuration_count
    )

    actual_per_query_rows = len(
        artifacts.per_query
    )

    if actual_per_query_rows != expected_per_query_rows:
        raise RuntimeError(
            "Số dòng per-query metrics không đúng.\n"
            f"Expected: {expected_per_query_rows}\n"
            f"Actual  : {actual_per_query_rows}"
        )

    if artifacts.overall.empty:
        raise RuntimeError(
            "Không tạo được overall metrics."
        )

    paths = save_evaluation_artifacts(
        artifacts=artifacts,
        output_dir=args.output_dir,
    )

    _print_overall_summary(
        artifacts.overall
    )

    best_row_index = (
        artifacts.overall["ndcg_at_5_mean"].idxmax()
    )

    best_row = artifacts.overall.loc[
        best_row_index
    ]

    print(
        "\nCấu hình có NDCG@5 trung bình cao nhất:"
        f"\n- {best_row['configuration']}"
        f"\n- NDCG@5 = "
        f"{best_row['ndcg_at_5_mean']:.4f}"
    )

    print("\n[✓] Đã tạo các file:")

    for name, path in paths.items():
        print(f"- {name:10s}: {path}")

    relevance_null_count = int(
        scored_predictions["relevance"]
        .isna()
        .sum()
    )

    print(
        "\nKiểm tra hoàn tất:"
        f"\n- per_query_metrics rows: "
        f"{actual_per_query_rows}"
        f"\n- overall_metrics rows  : "
        f"{len(artifacts.overall)}"
        f"\n- relevance null        : "
        f"{relevance_null_count}"
    )


if __name__ == "__main__":
    main()