from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

import src.api.app as app_module
from src.services.recommendation_service import (
    RecommendationResult,
)


class FakeRecommendationService:
    """
    Service giả dành cho API tests.

    Không load embedding model, FAISS index hoặc Cross-Encoder.
    """

    def __init__(
        self,
        retrieval_k: int,
        top_n: int,
    ) -> None:
        self.retrieval_k = retrieval_k
        self.top_n = top_n

        # Health endpoint chỉ kiểm tra thuộc tính này khác None.
        self.vector_store = object()

    def recommend(
        self,
        query: str,
        include_explanation: bool,
        configuration: str,
        top_n: int,
    ) -> RecommendationResult:
        recommendations: list[dict[str, Any]] = []

        for index in range(top_n):
            rank = index + 1

            recommendations.append(
                {
                    "movieId": 1000 + rank,
                    "title": f"Test Movie {rank}",
                    "genres": "Drama|Sci-Fi",
                    "rank": rank,
                    "retrieval_rank": rank,
                    "result_rank": rank,
                    "final_rank": rank,
                    "rating_mean": 4.0,
                    "rating_count": 1000,
                    "semantic_similarity": 0.90,
                    "popularity_score": 0.70,
                    "rule_score": 0.10,
                    "cross_encoder_score": (
                        0.80
                        if configuration == "hybrid_with_ce"
                        else None
                    ),
                    "evaluation_score": 0.85,
                    "final_score": 0.85,
                }
            )

        explanation = (
            "Test explanation."
            if include_explanation
            else ""
        )

        return RecommendationResult(
            query=query,
            configuration=configuration,
            recommendations=recommendations,
            explanation=explanation,
            timings={
                "ranking_seconds": 0.01,
                "generation_seconds": (
                    0.02
                    if include_explanation
                    else 0.0
                ),
                "total_seconds": (
                    0.03
                    if include_explanation
                    else 0.01
                ),
            },
        )


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    """
    Tạo TestClient và thay service thật bằng fake service.

    Lifespan vẫn chạy bình thường, nhưng không load ML models.
    """

    monkeypatch.setattr(
        app_module,
        "RecommendationService",
        FakeRecommendationService,
    )

    with TestClient(app_module.app) as test_client:
        yield test_client