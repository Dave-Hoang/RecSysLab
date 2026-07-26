from pathlib import Path

import pandas as pd

from src.config import (
    EVALUATION_CANDIDATES_FOR_LABELING_PATH as CANDIDATES_PATH,
    EVALUATION_LABELS_PATH as LABELS_PATH,
)

REQUIRED_COLUMNS = {
    "query_id",
    "query",
    "category",
    "difficulty",
    "movieId",
    "title",
    "genres",
    "relevance",
}


def build_labels() -> pd.DataFrame:
    if not CANDIDATES_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {CANDIDATES_PATH}"
        )

    candidates = pd.read_csv(
        CANDIDATES_PATH,
        dtype={
            "candidate_id": "string",
            "query_id": "string",
            "query": "string",
            "category": "string",
            "difficulty": "string",
            "title": "string",
            "genres": "string",
            "notes": "string",
        },
    )

    missing_columns = REQUIRED_COLUMNS.difference(
        candidates.columns
    )

    if missing_columns:
        raise ValueError(
            "candidates_for_labeling.csv thiếu các cột: "
            f"{sorted(missing_columns)}"
        )

    if candidates.empty:
        raise ValueError(
            "candidates_for_labeling.csv không có dữ liệu."
        )

    # Chuẩn hóa khóa merge
    candidates["query_id"] = (
        candidates["query_id"]
        .astype("string")
        .str.strip()
    )

    candidates["movieId"] = pd.to_numeric(
        candidates["movieId"],
        errors="raise",
    ).astype("int64")

    # Chuẩn hóa relevance
    candidates["relevance"] = pd.to_numeric(
        candidates["relevance"],
        errors="coerce",
    )

    missing_relevance = candidates[
        candidates["relevance"].isna()
    ]

    if not missing_relevance.empty:
        columns_to_show = [
            "query_id",
            "movieId",
            "title",
            "relevance",
        ]

        raise ValueError(
            "Vẫn còn movie chưa được gán relevance:\n"
            f"{missing_relevance[columns_to_show].to_string(index=False)}"
        )

    candidates["relevance"] = (
        candidates["relevance"].astype("int8")
    )

    invalid_relevance = candidates[
        ~candidates["relevance"].isin([0, 1, 2])
    ]

    if not invalid_relevance.empty:
        columns_to_show = [
            "query_id",
            "movieId",
            "title",
            "relevance",
        ]

        raise ValueError(
            "Relevance chỉ được nhận giá trị 0, 1 hoặc 2:\n"
            f"{invalid_relevance[columns_to_show].to_string(index=False)}"
        )

    # Một query-movie chỉ được có một ground-truth label
    duplicated_mask = candidates.duplicated(
        subset=["query_id", "movieId"],
        keep=False,
    )

    if duplicated_mask.any():
        duplicated_rows = candidates.loc[
            duplicated_mask,
            [
                "query_id",
                "movieId",
                "title",
                "relevance",
            ],
        ].sort_values(["query_id", "movieId"])

        raise ValueError(
            "Có cặp query_id + movieId bị trùng:\n"
            f"{duplicated_rows.to_string(index=False)}"
        )

    if "notes" not in candidates.columns:
        candidates["notes"] = ""

    candidates["notes"] = (
        candidates["notes"]
        .fillna("")
        .astype("string")
    )

    # Ground truth sạch dùng cho evaluation
    label_columns = [
        "query_id",
        "query",
        "category",
        "difficulty",
        "movieId",
        "title",
        "genres",
        "relevance",
        "notes",
    ]

    labels = (
        candidates[label_columns]
        .sort_values(
            by=["query_id", "movieId"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return labels


def main() -> None:
    print("=" * 60)
    print("BUILD HUMAN RELEVANCE LABELS")
    print("=" * 60)

    labels = build_labels()

    LABELS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels.to_csv(
        LABELS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[✓] Đã tạo: {LABELS_PATH}")
    print(f"[✓] Tổng số labels: {len(labels):,}")
    print(f"[✓] Số query: {labels['query_id'].nunique()}")

    print("\nPhân bố relevance:")

    relevance_counts = (
        labels["relevance"]
        .value_counts()
        .sort_index()
    )

    for relevance in [0, 1, 2]:
        count = int(relevance_counts.get(relevance, 0))
        print(f"- relevance={relevance}: {count}")

    print("\nKiểm tra:")
    print(
        "- Relevance còn trống:",
        int(labels["relevance"].isna().sum()),
    )
    print(
        "- Khóa query_id + movieId bị trùng:",
        int(
            labels.duplicated(
                ["query_id", "movieId"]
            ).sum()
        ),
    )


if __name__ == "__main__":
    main()