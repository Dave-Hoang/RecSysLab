from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import (
    get_production_settings,
    get_recommendation_service,
)
from src.api.schemas import HealthResponse
from src.production_config import ProductionSettings
from src.services.recommendation_service import (
    RecommendationService,
)


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get(
    "",
    response_model=HealthResponse,
    summary="Kiểm tra trạng thái API",
)
def health_check(
    settings: ProductionSettings = Depends(
        get_production_settings
    ),
    service: RecommendationService = Depends(
        get_recommendation_service
    ),
) -> HealthResponse:
    """Kiểm tra config, service và FAISS vector store."""

    return HealthResponse(
        status="ok",
        service="movie-recommendation-api",
        default_mode=settings.default_mode,
        available_modes=sorted(settings.modes),
        vector_store_loaded=(
            service.vector_store is not None
        ),
    )