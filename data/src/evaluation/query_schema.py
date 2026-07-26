from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.constants import (
    QueryCategory,
    QueryDifficulty,
)


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    category: QueryCategory
    difficulty: QueryDifficulty
    intent: str
    query: str
    expected_focus: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id không được để trống.")

        if not self.query.strip():
            raise ValueError(
                f"Query của {self.query_id} không được để trống."
            )

        if not self.intent.strip():
            raise ValueError(
                f"Intent của {self.query_id} không được để trống."
            )

        if not self.expected_focus:
            raise ValueError(
                f"expected_focus của {self.query_id} không được để trống."
            )