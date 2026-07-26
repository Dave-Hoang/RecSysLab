from __future__ import annotations

import csv
from pathlib import Path

from src.config import EVALUATION_QUERIES_PATH
from src.evaluation.constants import (
    QueryCategory,
    QueryDifficulty,
)
from src.evaluation.query_schema import EvaluationQuery


REQUIRED_COLUMNS = {
    "query_id",
    "category",
    "difficulty",
    "intent",
    "query",
    "expected_focus",
}


def _parse_expected_focus(raw_value: str) -> tuple[str, ...]:
    values = [
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    ]

    return tuple(values)


def load_evaluation_queries(
    file_path: Path = EVALUATION_QUERIES_PATH,
) -> list[EvaluationQuery]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy evaluation query file: {file_path}"
        )

    queries: list[EvaluationQuery] = []

    with file_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError("queries.csv không có header.")

        actual_columns = set(reader.fieldnames)
        missing_columns = REQUIRED_COLUMNS - actual_columns

        if missing_columns:
            raise ValueError(
                "queries.csv thiếu các cột: "
                f"{sorted(missing_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                evaluation_query = EvaluationQuery(
                    query_id=row["query_id"].strip(),
                    category=QueryCategory(
                        row["category"].strip()
                    ),
                    difficulty=QueryDifficulty(
                        row["difficulty"].strip()
                    ),
                    intent=row["intent"].strip(),
                    query=row["query"].strip(),
                    expected_focus=_parse_expected_focus(
                        row["expected_focus"]
                    ),
                )
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"Lỗi tại dòng {row_number}: {error}"
                ) from error

            queries.append(evaluation_query)

    _validate_unique_query_ids(queries)
    _validate_query_count(queries)

    return queries


def _validate_unique_query_ids(
    queries: list[EvaluationQuery],
) -> None:
    query_ids = [item.query_id for item in queries]

    duplicated_ids = {
        query_id
        for query_id in query_ids
        if query_ids.count(query_id) > 1
    }

    if duplicated_ids:
        raise ValueError(
            "Có query_id bị trùng: "
            f"{sorted(duplicated_ids)}"
        )


def _validate_query_count(
    queries: list[EvaluationQuery],
) -> None:
    expected_count = 30
    actual_count = len(queries)

    if actual_count != expected_count:
        raise ValueError(
            f"Expected {expected_count} queries, "
            f"nhưng tìm thấy {actual_count}."
        )