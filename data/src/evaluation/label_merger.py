from __future__ import annotations

from pathlib import Path

import pandas as pd


PREDICTION_REQUIRED_COLUMNS = {
    "query_id",
    "query",
    "category",
    "difficulty",
    "configuration",
    "rank",
    "movieId",
    "title",
}

LABEL_REQUIRED_COLUMNS = {
    "query_id",
    "movieId",
    "relevance",
}

EXPECTED_CONFIGURATIONS = {
    "faiss_only",
    "hybrid_no_ce",
    "cross_encoder_only",
    "hybrid_with_ce",
}

EXPECTED_QUERY_COUNT = 30
EVALUATION_TOP_K = 5


def _validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataframe_name: str,
) -> None:
    """Kiểm tra DataFrame có đầy đủ các cột bắt buộc."""

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{dataframe_name} thiếu các cột bắt buộc: "
            f"{sorted(missing_columns)}"
        )


def load_predictions(path: Path) -> pd.DataFrame:
    """Đọc và kiểm tra predictions.csv."""

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy predictions file: {path}"
        )

    predictions = pd.read_csv(
        path,
        dtype={
            "query_id": "string",
            "query": "string",
            "category": "string",
            "difficulty": "string",
            "configuration": "string",
            "title": "string",
            "genres": "string",
        },
    )

    _validate_required_columns(
        dataframe=predictions,
        required_columns=PREDICTION_REQUIRED_COLUMNS,
        dataframe_name="predictions.csv",
    )

    predictions["query_id"] = predictions["query_id"].str.strip()
    predictions["configuration"] = (
        predictions["configuration"].str.strip()
    )

    predictions["movieId"] = pd.to_numeric(
        predictions["movieId"],
        errors="raise",
    ).astype("int64")

    predictions["rank"] = pd.to_numeric(
        predictions["rank"],
        errors="raise",
    ).astype("int64")

    if predictions.empty:
        raise ValueError("predictions.csv không có dữ liệu.")

    return predictions


def load_labels(path: Path) -> pd.DataFrame:
    """Đọc và kiểm tra labels.csv."""

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy labels file: {path}"
        )

    labels = pd.read_csv(
        path,
        dtype={
            "query_id": "string",
            "notes": "string",
        },
    )

    _validate_required_columns(
        dataframe=labels,
        required_columns=LABEL_REQUIRED_COLUMNS,
        dataframe_name="labels.csv",
    )

    labels["query_id"] = labels["query_id"].str.strip()

    labels["movieId"] = pd.to_numeric(
        labels["movieId"],
        errors="raise",
    ).astype("int64")

    labels["relevance"] = pd.to_numeric(
        labels["relevance"],
        errors="coerce",
    )

    if labels["relevance"].isna().any():
        invalid_rows = labels.loc[
            labels["relevance"].isna(),
            ["query_id", "movieId", "relevance"],
        ]

        raise ValueError(
            "labels.csv còn relevance trống hoặc không hợp lệ:\n"
            f"{invalid_rows.to_string(index=False)}"
        )

    labels["relevance"] = labels["relevance"].astype("int8")

    invalid_relevance = labels.loc[
        ~labels["relevance"].isin([0, 1, 2]),
        ["query_id", "movieId", "relevance"],
    ]

    if not invalid_relevance.empty:
        raise ValueError(
            "Relevance chỉ được nhận giá trị 0, 1 hoặc 2:\n"
            f"{invalid_relevance.to_string(index=False)}"
        )

    if "notes" not in labels.columns:
        labels["notes"] = ""

    labels["notes"] = labels["notes"].fillna("").astype("string")

    return labels


def validate_predictions(predictions: pd.DataFrame) -> None:
    """Kiểm tra predictions có đúng thiết kế benchmark."""

    actual_configurations = set(
        predictions["configuration"].dropna().unique()
    )

    if actual_configurations != EXPECTED_CONFIGURATIONS:
        raise ValueError(
            "Danh sách configuration không đúng.\n"
            f"Expected: {sorted(EXPECTED_CONFIGURATIONS)}\n"
            f"Actual:   {sorted(actual_configurations)}"
        )

    query_count = predictions["query_id"].nunique()

    if query_count != EXPECTED_QUERY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_QUERY_COUNT} query nhưng "
            f"tìm thấy {query_count}."
        )

    expected_rows = (
        EXPECTED_QUERY_COUNT
        * len(EXPECTED_CONFIGURATIONS)
        * EVALUATION_TOP_K
    )

    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} prediction rows nhưng "
            f"tìm thấy {len(predictions)}."
        )

    invalid_ranks = predictions.loc[
        ~predictions["rank"].between(1, EVALUATION_TOP_K),
        [
            "query_id",
            "configuration",
            "movieId",
            "rank",
        ],
    ]

    if not invalid_ranks.empty:
        raise ValueError(
            "Prediction có rank nằm ngoài khoảng 1–5:\n"
            f"{invalid_ranks.to_string(index=False)}"
        )

    group_sizes = (
        predictions
        .groupby(
            ["query_id", "configuration"],
            observed=True,
        )
        .size()
    )

    invalid_groups = group_sizes.loc[
        group_sizes != EVALUATION_TOP_K
    ]

    if not invalid_groups.empty:
        raise ValueError(
            "Một số query × configuration không có đúng "
            f"{EVALUATION_TOP_K} kết quả:\n"
            f"{invalid_groups.to_string()}"
        )

    duplicated_ranks = predictions.duplicated(
        subset=["query_id", "configuration", "rank"],
        keep=False,
    )

    if duplicated_ranks.any():
        duplicate_rows = predictions.loc[
            duplicated_ranks,
            [
                "query_id",
                "configuration",
                "rank",
                "movieId",
                "title",
            ],
        ]

        raise ValueError(
            "Có rank bị trùng trong cùng query và configuration:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    duplicated_movies = predictions.duplicated(
        subset=["query_id", "configuration", "movieId"],
        keep=False,
    )

    if duplicated_movies.any():
        duplicate_rows = predictions.loc[
            duplicated_movies,
            [
                "query_id",
                "configuration",
                "rank",
                "movieId",
                "title",
            ],
        ]

        raise ValueError(
            "Một phim xuất hiện nhiều lần trong cùng một "
            "query và configuration:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )


def validate_labels(labels: pd.DataFrame) -> None:
    """Kiểm tra mỗi query và movie chỉ có một nhãn."""

    duplicated_labels = labels.duplicated(
        subset=["query_id", "movieId"],
        keep=False,
    )

    if duplicated_labels.any():
        duplicate_rows = labels.loc[
            duplicated_labels,
            [
                "query_id",
                "movieId",
                "relevance",
            ],
        ].sort_values(["query_id", "movieId"])

        raise ValueError(
            "labels.csv có cặp query_id + movieId bị trùng:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )


def validate_label_coverage(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    """
    Kiểm tra tất cả prediction đều có label và label không bị mồ côi.
    """

    prediction_keys = predictions[
        ["query_id", "movieId"]
    ].drop_duplicates()

    label_keys = labels[
        ["query_id", "movieId"]
    ].drop_duplicates()

    missing_labels = prediction_keys.merge(
        label_keys,
        on=["query_id", "movieId"],
        how="left",
        indicator=True,
    )

    missing_labels = missing_labels.loc[
        missing_labels["_merge"] == "left_only",
        ["query_id", "movieId"],
    ]

    if not missing_labels.empty:
        raise ValueError(
            "Có prediction chưa được gán relevance:\n"
            f"{missing_labels.to_string(index=False)}"
        )

    orphan_labels = label_keys.merge(
        prediction_keys,
        on=["query_id", "movieId"],
        how="left",
        indicator=True,
    )

    orphan_labels = orphan_labels.loc[
        orphan_labels["_merge"] == "left_only",
        ["query_id", "movieId"],
    ]

    if not orphan_labels.empty:
        raise ValueError(
            "Có label không xuất hiện trong predictions.csv:\n"
            f"{orphan_labels.to_string(index=False)}"
        )


def merge_predictions_with_labels(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ghép relevance vào từng prediction.

    Quan hệ:
        nhiều prediction rows
        → một label cho mỗi query_id + movieId
    """

    validate_predictions(predictions)
    validate_labels(labels)
    validate_label_coverage(predictions, labels)

    label_columns = [
        "query_id",
        "movieId",
        "relevance",
        "notes",
    ]

    row_count_before = len(predictions)

    scored_predictions = predictions.merge(
        labels[label_columns],
        on=["query_id", "movieId"],
        how="left",
        validate="many_to_one",
    )

    if len(scored_predictions) != row_count_before:
        raise RuntimeError(
            "Số dòng đã thay đổi sau khi merge. "
            f"Trước merge: {row_count_before}, "
            f"sau merge: {len(scored_predictions)}."
        )

    if scored_predictions["relevance"].isna().any():
        missing_rows = scored_predictions.loc[
            scored_predictions["relevance"].isna(),
            [
                "query_id",
                "configuration",
                "movieId",
                "title",
            ],
        ]

        raise RuntimeError(
            "Một số prediction không có relevance sau merge:\n"
            f"{missing_rows.to_string(index=False)}"
        )

    scored_predictions["relevance"] = (
        scored_predictions["relevance"].astype("int8")
    )

    scored_predictions = scored_predictions.sort_values(
        by=["query_id", "configuration", "rank"],
        kind="stable",
    ).reset_index(drop=True)

    return scored_predictions


def save_scored_predictions(
    scored_predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Lưu kết quả an toàn, không ghi đè trực tiếp giữa chừng."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(".tmp.csv")

    scored_predictions.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8-sig",
    )

    temporary_path.replace(output_path)


def run_label_merge(
    predictions_path: Path,
    labels_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Chạy toàn bộ pipeline merge của giai đoạn 3."""

    predictions = load_predictions(predictions_path)
    labels = load_labels(labels_path)

    scored_predictions = merge_predictions_with_labels(
        predictions=predictions,
        labels=labels,
    )

    save_scored_predictions(
        scored_predictions=scored_predictions,
        output_path=output_path,
    )

    return scored_predictions