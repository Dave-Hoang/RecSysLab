from __future__ import annotations

import logging

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
import pandas as pd

from src.api.dependencies import (
    get_production_settings,
    get_recommendation_service,
)
from src.api.schemas import (
    ErrorResponse,
    RecommendationRequest,
    AgenticRecommendationResponse,
)
from src.production_config import ProductionSettings
from src.services.recommendation_service import (
    RecommendationService,
)
from src.graph.graph import build_graph
from src.ranking.cross_encoder import get_cross_encoder


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/recommendations/agentic",
    tags=["Agentic Recommendations"],
)


@router.post(
    "",
    response_model=AgenticRecommendationResponse,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Request không hợp lệ.",
        },
        503: {
            "model": ErrorResponse,
            "description": (
                "Recommendation service hoặc model không sẵn sàng."
            ),
        },
    },
    summary="Đề xuất phim bằng LangGraph Agent",
)
def agentic_recommend_movies(
    payload: RecommendationRequest,
    service: RecommendationService = Depends(
        get_recommendation_service
    ),
    settings: ProductionSettings = Depends(
        get_production_settings
    ),
) -> AgenticRecommendationResponse:
    """
    Chạy recommendation pipeline sử dụng kiến trúc LangGraph StateGraph (Agentic).
    """

    try:
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
        # Load compiled graph using dependencies
        compiled_graph = build_graph(
            vector_store=service.vector_store,
            cross_encoder=get_cross_encoder(),
        )

        initial_state = {
            "original_query": payload.query,
            "top_n": selected_top_k,
            "include_explanation": include_explanation,
        }

        # Invoke graph
        result = compiled_graph.invoke(initial_state)
        
        # Convert dataframe to JSON records using service's method if ranked_movies is present
        recommendations = []
        if "ranked_movies" in result and isinstance(result["ranked_movies"], pd.DataFrame):
            recommendations = service._dataframe_to_records(result["ranked_movies"])
            
            # For explanations
            if "explanations" in result and len(result["explanations"]) == len(recommendations):
                for idx, exp in enumerate(result["explanations"]):
                    recommendations[idx]["explanation"] = exp

    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except Exception as error:
        logger.exception(
            "Agentic Recommendation pipeline gặp lỗi ngoài dự kiến."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="Internal agentic recommendation error.",
        ) from error

    return AgenticRecommendationResponse(
        query=payload.query,
        top_k=selected_top_k,
        intent=result.get("intent"),
        expanded_query=result.get("expanded_query"),
        confidence_level=result.get("confidence_level"),
        execution_path=result.get("execution_path", []),
        recommendations=recommendations,
        direct_response=result.get("direct_response"),
        timings=result.get("timings", {}),
    )
