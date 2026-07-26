from __future__ import annotations

from fastapi import HTTPException, Request, status

from src.production_config import ProductionSettings
from src.services.recommendation_service import (
    RecommendationService,
)


def get_production_settings(
    request: Request,
) -> ProductionSettings:
    """Lấy production settings đã được load lúc startup."""

    settings = getattr(
        request.app.state,
        "production_settings",
        None,
    )

    if settings is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Production settings chưa sẵn sàng.",
        )

    return settings


def get_recommendation_service(
    request: Request,
) -> RecommendationService:
    """Lấy RecommendationService singleton từ application state."""

    service = getattr(
        request.app.state,
        "recommendation_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service chưa sẵn sàng.",
        )

    return service