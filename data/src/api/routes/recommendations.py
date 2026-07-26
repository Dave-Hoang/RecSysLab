from __future__ import annotations

import logging

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from src.api.dependencies import (
    get_production_settings,
    get_recommendation_service,
)
from src.api.schemas import (
    ErrorResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from src.production_config import ProductionSettings
from src.services.recommendation_service import (
    RecommendationService,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/recommendations",
    tags=["Recommendations"],
)


@router.post(
    "",
    response_model=RecommendationResponse,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Request không hợp lệ.",
        },
        503: {
            "model": ErrorResponse,
            "description": (
                "Recommendation service hoặc model "
                "không sẵn sàng."
            ),
        },
    },
    summary="Đề xuất phim từ truy vấn ngôn ngữ tự nhiên",
)
def recommend_movies(
    payload: RecommendationRequest,
    service: RecommendationService = Depends(
        get_recommendation_service
    ),
    settings: ProductionSettings = Depends(
        get_production_settings
    ),
) -> RecommendationResponse:
    """
    Chạy recommendation theo production mode.

    quality:
        hybrid_with_ce

    fast:
        hybrid_no_ce
    """

    try:
        selected_mode = settings.get_mode(
            payload.mode
        )

        selected_top_k = settings.validate_top_k(
            payload.top_k
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    include_explanation = (
        settings.default_include_explanation
        if payload.include_explanation is None
        else payload.include_explanation
    )

    try:
        result = service.recommend(
            query=payload.query,
            configuration=(
                selected_mode.configuration
            ),
            top_n=selected_top_k,
            include_explanation=include_explanation,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except RuntimeError as error:
        logger.exception(
            "Recommendation pipeline gặp lỗi runtime."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Không thể hoàn thành recommendation "
                "do model hoặc dữ liệu chưa sẵn sàng."
            ),
        ) from error
    except Exception as error:
        logger.exception(
            "Recommendation pipeline gặp lỗi ngoài dự kiến."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="Internal recommendation error.",
        ) from error

    return RecommendationResponse(
        query=result.query,
        mode=selected_mode.name,
        configuration=result.configuration,
        top_k=len(result.recommendations),
        recommendations=result.recommendations,
        timings=result.timings,
    )