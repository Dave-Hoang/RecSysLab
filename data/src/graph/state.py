"""
RecSysState — Trạng thái chia sẻ giữa tất cả các Node trong LangGraph.

Mỗi node đọc input từ state và ghi output vào state.
TypedDict với total=False cho phép các field là optional
(chỉ được set bởi node tương ứng).
"""

from typing import TypedDict, Optional

import pandas as pd


class RecSysState(TypedDict, total=False):
    """Trạng thái xuyên suốt Graph."""

    # ── Input ──
    original_query: str                  # Query gốc từ người dùng
    include_explanation: bool            # Có cần LLM explanation không
    top_n: int                           # Số kết quả trả về

    # ── Node 1: Query Analyzer output ──
    intent: str                          # "search_movie" | "need_expansion" | "not_recommendation"
    intent_reasoning: str                # Lý do phân loại (cho debug/trace)
    extracted_entities: dict             # {"genres": [...], "emotions": [...], "constraints": [...]}

    # ── Node 3: Query Expansion output ──
    expanded_query: str                  # Query đã được viết lại / mở rộng
    previous_expanded_query: str         # Query expansion từ lần retry trước
    search_query: str                    # Query cuối cùng dùng cho FAISS (= original hoặc expanded)

    # ── Node 2 & 5: Retrieval / Reranking output ──
    ranked_movies: pd.DataFrame          # DataFrame kết quả (từ Node 2 hoặc Node 5)
    used_cross_encoder: bool             # True nếu đi qua Node 5, False nếu chỉ Node 2

    # ── Node 6: Quality Gate output ──
    mean_cross_encoder_score: Optional[float]  # Trung bình CE score (chỉ có khi qua Node 5)
    mean_semantic_score: float           # Trung bình Semantic score (luôn có)
    confidence_level: str                # "high" | "low"
    retry_count: int                     # Số lần đã retry (tối đa 1)

    # ── Node 7: Explanation output ──
    explanations: list[str]              # Giải thích cho từng phim

    # ── Node 4: Direct Response output ──
    direct_response: str                 # Câu trả lời trực tiếp (chitchat/clarification)

    # ── Metadata ──
    execution_path: list[str]            # Danh sách các node đã đi qua
    timings: dict[str, float]            # Thời gian chạy từng node
