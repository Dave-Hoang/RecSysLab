from src.config import (
    EVALUATION_LABELS_PATH,
    EVALUATION_PREDICTIONS_PATH,
    EVALUATION_SCORED_PREDICTIONS_PATH,
)
from src.evaluation.label_merger import run_label_merge


def main() -> None:
    print("=" * 65)
    print("MERGE EVALUATION PREDICTIONS WITH HUMAN LABELS")
    print("=" * 65)

    print(f"Predictions: {EVALUATION_PREDICTIONS_PATH}")
    print(f"Labels     : {EVALUATION_LABELS_PATH}")
    print(f"Output     : {EVALUATION_SCORED_PREDICTIONS_PATH}")

    scored_predictions = run_label_merge(
        predictions_path=EVALUATION_PREDICTIONS_PATH,
        labels_path=EVALUATION_LABELS_PATH,
        output_path=EVALUATION_SCORED_PREDICTIONS_PATH,
    )

    print("\n[✓] Merge thành công.")
    print(f"[✓] Tổng số rows: {len(scored_predictions):,}")
    print(
        f"[✓] Số query: "
        f"{scored_predictions['query_id'].nunique()}"
    )
    print(
        f"[✓] Số configuration: "
        f"{scored_predictions['configuration'].nunique()}"
    )

    print("\nSố rows theo configuration:")
    configuration_counts = (
        scored_predictions
        .groupby("configuration", observed=True)
        .size()
        .sort_index()
    )

    for configuration, count in configuration_counts.items():
        print(f"- {configuration}: {count}")

    print("\nPhân bố relevance trên prediction rows:")
    relevance_counts = (
        scored_predictions["relevance"]
        .value_counts()
        .sort_index()
    )

    for relevance in [0, 1, 2]:
        count = int(relevance_counts.get(relevance, 0))
        print(f"- relevance={relevance}: {count}")

    print(
        "\n[✓] Đã lưu scored predictions tại:\n"
        f"    {EVALUATION_SCORED_PREDICTIONS_PATH}"
    )


if __name__ == "__main__":
    main()