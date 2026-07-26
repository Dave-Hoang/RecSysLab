from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    EVALUATION_CANDIDATES_FOR_LABELING_PATH,
    EVALUATION_PREDICTIONS_PATH,
)


# Một phim chỉ bị xem là trùng khi xuất hiện nhiều lần
# trong cùng một evaluation query.
UNIQUE_KEY_COLUMNS = [
    "query_id",
    "movieId",
]

# Chỉ giữ các cột trung lập cần thiết cho human labeling.
# Không giữ configuration, rank hoặc score để tránh bias.
LABELING_COLUMNS = [
    "query_id",
    "query",
    "category",
    "difficulty",
    "movieId",
    "title",
    "genres",
]

RANDOM_SEED = 42


def validate_predictions(predictions: pd.DataFrame) -> None:
    """Kiểm tra schema tối thiểu của predictions.csv."""

    required_columns = set(
        UNIQUE_KEY_COLUMNS + LABELING_COLUMNS
    )

    missing_columns = required_columns.difference(
        predictions.columns
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )
        raise ValueError(
            "predictions.csv thiếu các cột bắt buộc: "
            f"{missing_text}"
        )

    if predictions.empty:
        raise ValueError(
            "predictions.csv không chứa dữ liệu."
        )


def normalize_keys(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Chuẩn hóa query_id và movieId trước khi loại trùng."""

    normalized = predictions.copy()

    normalized["query_id"] = (
        normalized["query_id"]
        .astype("string")
        .str.strip()
    )

    if (
        normalized["query_id"].isna().any()
        or normalized["query_id"].eq("").any()
    ):
        raise ValueError(
            "Phát hiện query_id rỗng hoặc không hợp lệ."
        )

    normalized["movieId"] = pd.to_numeric(
        normalized["movieId"],
        errors="raise",
    ).astype("int64")

    return normalized


def validate_metadata_consistency(
    predictions: pd.DataFrame,
) -> None:
    """
    Đảm bảo cùng một query_id + movieId không có metadata
    khác nhau giữa các configuration.
    """

    metadata_columns = [
        "query",
        "category",
        "difficulty",
        "title",
        "genres",
    ]

    unique_counts = (
        predictions
        .groupby(
            UNIQUE_KEY_COLUMNS,
            dropna=False,
        )[metadata_columns]
        .nunique(dropna=False)
    )

    inconsistent_mask = unique_counts.gt(1)

    if not inconsistent_mask.any().any():
        return

    inconsistent_keys = (
        inconsistent_mask
        .any(axis=1)
        .loc[lambda series: series]
        .index
        .tolist()
    )

    preview = inconsistent_keys[:5]

    raise ValueError(
        "Metadata không nhất quán giữa các configuration "
        "cho cùng query_id + movieId. "
        f"Ví dụ: {preview}"
    )


def build_labeling_candidates(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Lấy union phim của tất cả configuration theo từng query.

    Mỗi cặp query_id + movieId chỉ xuất hiện đúng một lần.
    """

    validate_predictions(predictions)

    normalized = normalize_keys(predictions)

    # Loại những dòng giống nhau hoàn toàn trước.
    normalized = (
        normalized
        .drop_duplicates()
        .reset_index(drop=True)
    )

    validate_metadata_consistency(normalized)

    # Vì metadata của các dòng trùng nhau đã được kiểm tra
    # là nhất quán, có thể giữ dòng đầu tiên.
    candidates = (
        normalized
        .sort_values(
            by=["query_id", "movieId"],
            kind="stable",
        )
        .drop_duplicates(
            subset=UNIQUE_KEY_COLUMNS,
            keep="first",
        )
        .loc[:, LABELING_COLUMNS]
        .copy()
    )

    # Trộn thứ tự phim trong từng query để người chấm
    # không nhìn thấy thứ tự do một configuration tạo ra.
    random_generator = np.random.default_rng(
        RANDOM_SEED
    )

    candidates["_random_order"] = (
        random_generator.random(len(candidates))
    )

    candidates = (
        candidates
        .sort_values(
            by=["query_id", "_random_order"],
            kind="stable",
        )
        .drop(columns="_random_order")
        .reset_index(drop=True)
    )

    # Đánh số candidate riêng trong mỗi query.
    candidates["candidate_order"] = (
        candidates
        .groupby(
            "query_id",
            sort=False,
        )
        .cumcount()
        .add(1)
    )

    # ID ổn định, thuận tiện cho kiểm tra và labeling.
    candidates["candidate_id"] = (
        candidates["query_id"].astype(str)
        + "_C"
        + candidates["candidate_order"]
        .astype(str)
        .str.zfill(2)
    )

    # Hai cột được người đánh giá điền thủ công.
    candidates["relevance"] = ""
    candidates["notes"] = ""

    final_columns = [
        "candidate_id",
        "query_id",
        "query",
        "category",
        "difficulty",
        "candidate_order",
        "movieId",
        "title",
        "genres",
        "relevance",
        "notes",
    ]

    candidates = candidates.loc[:, final_columns]

    remaining_duplicates = candidates.duplicated(
        subset=UNIQUE_KEY_COLUMNS
    ).sum()

    if remaining_duplicates != 0:
        raise RuntimeError(
            "Vẫn còn candidate trùng sau khi xử lý: "
            f"{remaining_duplicates}"
        )

    return candidates


def main() -> None:
    print(
        "[*] Đang đọc predictions từ:"
    )
    print(
        f"    {EVALUATION_PREDICTIONS_PATH}"
    )

    predictions = pd.read_csv(
        EVALUATION_PREDICTIONS_PATH
    )

    original_row_count = len(predictions)

    candidates = build_labeling_candidates(
        predictions
    )

    EVALUATION_CANDIDATES_FOR_LABELING_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # utf-8-sig giúp mở tiếng Việt đúng trong Excel.
    candidates.to_csv(
        EVALUATION_CANDIDATES_FOR_LABELING_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    removed_count = (
        original_row_count - len(candidates)
    )

    candidates_per_query = (
        candidates
        .groupby("query_id")
        .size()
    )

    print()
    print("[✓] Tạo candidates_for_labeling.csv thành công.")
    print(
        f"[✓] Số dòng predictions ban đầu: "
        f"{original_row_count}"
    )
    print(
        f"[✓] Số candidate độc bản: "
        f"{len(candidates)}"
    )
    print(
        f"[✓] Số lần xuất hiện trùng đã loại: "
        f"{removed_count}"
    )
    print(
        f"[✓] Số evaluation queries: "
        f"{candidates['query_id'].nunique()}"
    )
    print(
        f"[✓] Candidate ít nhất/query: "
        f"{candidates_per_query.min()}"
    )
    print(
        f"[✓] Candidate nhiều nhất/query: "
        f"{candidates_per_query.max()}"
    )
    print(
        "[✓] Số khóa query_id + movieId còn trùng: "
        f"{candidates.duplicated(UNIQUE_KEY_COLUMNS).sum()}"
    )
    print(
        f"[✓] File output: "
        f"{EVALUATION_CANDIDATES_FOR_LABELING_PATH}"
    )


if __name__ == "__main__":
    main()