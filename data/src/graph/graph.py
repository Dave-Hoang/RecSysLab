"""
LangGraph StateGraph builder & compiler.

Phase 1: Graph đơn giản START → fast_retrieval → END.
Phase 2: Thêm Node 1 (analyzer) + conditional edge route_by_intent().
Phase 3: Thêm Node 3 (expand), Node 4 (direct_response), Node 5 (full_reranking).
Phase 4: Thêm Node 6 (quality_gate) + conditional edge route_by_confidence()
         + Adaptive Escalation & retry loop.

Phase 5 Graph:
    START → analyzer → (route_by_intent)
              ├─ "search_movie"       → fast_retrieval ──┐
              ├─ "need_expansion"     → expand → full_reranking ──┤
              └─ "not_recommendation" → direct_response → END
                                                          │
              quality_gate ← ────────────────────────────┘
                  ├─ "high" → explanation → END
                  └─ "low"  → expand → full_reranking → quality_gate (retry loop)
"""

from functools import partial

from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from sentence_transformers import CrossEncoder

from src.graph.state import RecSysState
from src.graph.nodes import (
    analyzer_node,
    fast_retrieval_node,
    expansion_node,
    direct_response_node,
    full_reranking_node,
    full_reranking_node,
    quality_gate_node,
    explanation_node,
)


# ============================================================
# CONDITIONAL EDGE: ROUTE BY INTENT
# ============================================================

def route_by_intent(state: RecSysState) -> str:
    """
    Conditional edge sau Node 1 (Query Analyzer).

    Đọc `intent` từ state để quyết định node tiếp theo:
    - "search_movie"       → "fast_retrieval"  (Node 2, Fast Path)
    - "need_expansion"     → "expand"          (Node 3, Quality Path)
    - "not_recommendation" → "direct_response" (Node 4, Chitchat)

    Args:
        state: Trạng thái hiện tại, cần có field `intent`.

    Returns:
        Tên node tiếp theo dưới dạng string.
    """
    intent = state.get("intent", "search_movie")

    if intent == "search_movie":
        return "fast_retrieval"
    elif intent == "need_expansion":
        return "expand"
    else:
        # "not_recommendation"
        return "direct_response"


# ============================================================
# CONDITIONAL EDGE: ROUTE BY CONFIDENCE (Quality Gate)
# ============================================================

def route_by_confidence(state: RecSysState) -> str:
    """
    Conditional edge sau Node 6 (Quality Gate).

    Đọc `confidence_level` từ state để quyết định:
    - "high" → "explanation" (kết quả đủ tốt, hoặc đã hết retry)
    - "low"  → "expand" (quay lại Node 3 để retry với query mới)

    Lưu ý: Khi confidence = "low", Quality Gate đã tăng retry_count.
    MAX_RETRIES guard nằm trong quality_gate_node, nên ở đây chỉ cần
    đọc confidence_level.

    Args:
        state: Trạng thái hiện tại, cần có field `confidence_level`.

    Returns:
        "explanation" hoặc "expand".
    """
    if state.get("confidence_level") == "high":
        return "explanation"
    else:
        return "expand"


# ============================================================
# GRAPH BUILDER
# ============================================================

def build_graph(
    vector_store: FAISS,
    cross_encoder: CrossEncoder | None = None,
) -> StateGraph:
    """
    Build và compile LangGraph StateGraph cho RecSys.

    Phase 5 Graph:
        START → analyzer → (route_by_intent)
                  ├─ "search_movie"       → fast_retrieval ──┐
                  ├─ "need_expansion"     → expand → full_reranking ──┤
                  └─ "not_recommendation" → direct_response → END
                                                              │
                  quality_gate ← ────────────────────────────┘
                      ├─ "high" → explanation → END
                      └─ "low"  → expand → full_reranking → quality_gate (retry loop)

    Args:
        vector_store:
            FAISS vector store đã load, được inject vào node
            thông qua functools.partial.
        cross_encoder:
            Cross-Encoder model instance, inject vào full_reranking_node.
            Có thể None nếu chỉ chạy Fast Path.

    Returns:
        Compiled StateGraph, sẵn sàng gọi .invoke().
    """
    graph = StateGraph(RecSysState)

    # ── Bind external dependencies vào nodes qua partial ──
    # vector_store và cross_encoder quá lớn để serialize vào state,
    # nên inject qua functools.partial.
    bound_fast_retrieval = partial(
        fast_retrieval_node,
        vector_store=vector_store,
    )

    bound_full_reranking = partial(
        full_reranking_node,
        vector_store=vector_store,
        cross_encoder=cross_encoder,
    )

    # ── Add nodes ──
    graph.add_node("analyzer", analyzer_node)                # Node 1
    graph.add_node("fast_retrieval", bound_fast_retrieval)    # Node 2
    graph.add_node("expand", expansion_node)                 # Node 3
    graph.add_node("direct_response", direct_response_node)  # Node 4
    graph.add_node("full_reranking", bound_full_reranking)    # Node 5
    graph.add_node("quality_gate", quality_gate_node)         # Node 6
    graph.add_node("explanation", explanation_node)           # Node 7

    # ── Add edges ──
    # START → analyzer (always)
    graph.add_edge(START, "analyzer")

    # analyzer → conditional routing
    graph.add_conditional_edges(
        "analyzer",
        route_by_intent,
        {
            "fast_retrieval": "fast_retrieval",
            "expand": "expand",
            "direct_response": "direct_response",
        },
    )

    # Fast Path: fast_retrieval → quality_gate
    graph.add_edge("fast_retrieval", "quality_gate")

    # Quality Path: expand → full_reranking → quality_gate
    graph.add_edge("expand", "full_reranking")
    graph.add_edge("full_reranking", "quality_gate")

    # Quality Gate → conditional routing (retry loop)
    graph.add_conditional_edges(
        "quality_gate",
        route_by_confidence,
        {
            "explanation": "explanation", # confidence = "high" → explanation
            "expand": "expand",           # confidence = "low"  → retry qua Node 3
        },
    )

    # Explanation: explanation → END
    graph.add_edge("explanation", END)

    # Direct Response: direct_response → END
    graph.add_edge("direct_response", END)

    # ── Compile ──
    return graph.compile()
