from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes.health import router as health_router
from src.api.routes.recommendations import (
    router as recommendations_router,
)
from src.production_config import (
    load_production_settings,
)
from src.services.recommendation_service import (
    RecommendationService,
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """
    Khởi tạo resource nặng một lần khi API startup.

    Startup:
        production.yaml
        → embedding model
        → FAISS vector store
        → RecommendationService

    Shutdown:
        giải phóng reference trong app.state
    """

    logger.info("Đang load production settings...")

    production_settings = load_production_settings()

    logger.info("Đang khởi tạo RecommendationService...")

    recommendation_service = RecommendationService(
        retrieval_k=(
            production_settings.retrieval.candidate_k
        ),
        top_n=(
            production_settings.retrieval.default_top_k
        ),
    )

    app.state.production_settings = (
        production_settings
    )
    app.state.recommendation_service = (
        recommendation_service
    )

    logger.info(
        "Movie recommendation API đã sẵn sàng."
    )

    yield

    logger.info(
        "Đang shutdown movie recommendation API."
    )

    app.state.recommendation_service = None
    app.state.production_settings = None


app = FastAPI(
    title="RecSysLab Movie Recommendation API",
    description=(
        "Natural-language movie recommendation API "
        "using FAISS retrieval, hybrid ranking and "
        "Cross-Encoder reranking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(
    health_router,
)

app.include_router(
    recommendations_router,
)

@app.get(
    "/",
    include_in_schema=False,
)
def root() -> dict[str, str]:
    """Thông tin cơ bản của API."""

    return {
        "service": "RecSysLab Movie Recommendation API",
        "documentation": "/docs",
        "alternative_documentation": "/redoc",
        "health": "/health",
    }