from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class HealthResponse(BaseModel):
    """Response kiểm tra trạng thái API."""

    status: str
    service: str
    default_mode: str
    available_modes: list[str]
    vector_store_loaded: bool


class RecommendationRequest(BaseModel):
    """Request body cho recommendation endpoint."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "query": "psychological sci-fi movies",
                    "mode": "quality",
                    "top_k": 5,
                    "include_explanation": False,
                },
                {
                    "query": "romantic comedy movies",
                    "mode": "fast",
                    "top_k": 5,
                    "include_explanation": False,
                },
            ]
        },
    )

    query: str = Field(
        min_length=2,
        max_length=500,
        description="Natural-language movie query.",
    )

    mode: str | None = Field(
        default=None,
        description=(
            "Production mode: quality hoặc fast. "
            "Nếu bỏ trống sẽ dùng default_mode."
        ),
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Số lượng phim trả về. "
            "Giới hạn tối đa lấy từ production.yaml."
        ),
    )

    include_explanation: bool | None = Field(
        default=None,
        description=(
            "Có gọi LLM để tạo phần giải thích hay không."
        ),
    )

    @field_validator("query")
    @classmethod
    def validate_query(
        cls,
        value: str,
    ) -> str:
        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                "Query không được để trống."
            )

        return cleaned_value

    @field_validator("mode")
    @classmethod
    def normalize_mode(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        cleaned_value = value.strip().lower()

        if not cleaned_value:
            return None

        return cleaned_value


class MovieRecommendation(BaseModel):
    """
    Một phim trong kết quả recommendation.

    extra='allow' giữ lại các score bổ sung
    mà ranking pipeline trả về.
    """

    model_config = ConfigDict(
        extra="allow",
    )

    movieId: int | None = None
    title: str
    genres: str | None = None

    explanation: str | None = None

    rank: int | None = None
    retrieval_rank: int | None = None
    result_rank: int | None = None
    final_rank: int | None = None

    rating_mean: float | None = None
    rating_count: float | None = None

    faiss_distance: float | None = None
    semantic_similarity: float | None = None
    popularity_score: float | None = None
    rule_score: float | None = None
    cross_encoder_score: float | None = None
    evaluation_score: float | None = None
    final_score: float | None = None


class TimingResponse(BaseModel):
    """Thời gian chạy của recommendation pipeline."""

    ranking_seconds: float
    generation_seconds: float
    total_seconds: float


class RecommendationResponse(BaseModel):
    """Response cuối cùng của recommendation endpoint."""

    query: str
    mode: str
    configuration: str
    top_k: int

    recommendations: list[MovieRecommendation]

    timings: TimingResponse


class ErrorResponse(BaseModel):
    """Schema chung cho response lỗi."""

    detail: str | dict[str, Any]
