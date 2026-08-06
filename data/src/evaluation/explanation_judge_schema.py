from pydantic import BaseModel, Field, computed_field


class ExplanationJudgeResult(BaseModel):
    """
    Schema kết quả đánh giá từ LLM Judge.
    """

    claims: list[str] = Field(
        default_factory=list,
        description="Danh sách các khẳng định thực tế trích xuất từ câu giải thích.",
    )
    faithfulness_score: float = Field(
        description="Điểm độ trung thực dữ liệu (0.0 đến 1.0).",
    )
    faithfulness_reason: str = Field(
        description="Giải thích bằng tiếng Việt cho điểm faithfulness_score.",
    )
    context_relevance_score: float = Field(
        description="Điểm độ liên quan ngữ cảnh và bào chữa thứ hạng (0.0 đến 1.0).",
    )
    context_relevance_reason: str = Field(
        description="Giải thích bằng tiếng Việt cho điểm context_relevance_score.",
    )

    @computed_field
    @property
    def is_hallucinated(self) -> bool:
        """
        Computed field: is_hallucinated = True nếu faithfulness_score < 0.8.
        """
        return self.faithfulness_score < 0.8
