from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationPrediction:
    query_id: str
    query: str
    category: str
    difficulty: str
    configuration: str
    rank: int
    movie_id: int
    title: str
    genres: str
    faiss_squared_l2_distance: float | None
    cosine_similarity_raw: float | None
    semantic_score_relative: float | None
    popularity_score: float | None
    rule_score: float | None
    cross_encoder_score: float | None
    evaluation_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)