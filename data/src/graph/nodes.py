"""
Node functions cho LangGraph RecSys Graph.

Phase 1: fast_retrieval_node (Node 2 — Fast Path).
Phase 2: analyzer_node (Node 1 — Query Analyzer & Router).
Phase 3: expansion_node (Node 3), direct_response_node (Node 4),
         full_reranking_node (Node 5).
"""

import json
import time
import os

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from src.config import (
    ENV_PATH,
    GEMINI_MODEL_NAME,
    GOOGLE_API_KEY_ENV_NAME,
    LLM_TEMPERATURE,
    GROQ_API_KEY_ENV_NAME,
    GROQ_ROUTER_MODEL,
    GROQ_EXPANSION_MODEL,
    USE_GROQ_FOR_ROUTER,
)
from src.graph.config import (
    CE_CONFIDENCE_THRESHOLD,
    FINAL_RECOMMENDATION_TOP_K,
    MAX_RETRIES,
    RETRIEVAL_TOP_K,
    SEMANTIC_CONFIDENCE_THRESHOLD,
)
from src.graph.prompts import (
    ROUTER_SYSTEM_PROMPT,
    ROUTER_USER_TEMPLATE,
    EXPANSION_SYSTEM_PROMPT,
    EXPANSION_USER_TEMPLATE,
    EXPANSION_RETRY_PROMPT,
    DIRECT_RESPONSE_SYSTEM_PROMPT,
    DIRECT_RESPONSE_USER_TEMPLATE,
)
from src.graph.state import RecSysState
from src.ranking.hybrid_ranker import rank_without_ce, rank_with_ce
from src.retrieval.retriever import retrieve_movies_with_score
from src.generation.explanation_chain import explain_ranked_movies


# ============================================================
# LLM INITIALIZATION (lazy singleton — khởi tạo khi cần)
# ============================================================

_llm_gemini: ChatGoogleGenerativeAI | None = None
_llm_groq_router: ChatGroq | None = None
_llm_groq_expansion: ChatGroq | None = None


def _get_gemini_llm() -> ChatGoogleGenerativeAI:
    """Trả về singleton Gemini LLM instance."""
    global _llm_gemini
    if _llm_gemini is None:
        load_dotenv(ENV_PATH)
        _llm_gemini = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
        )
    return _llm_gemini


def _get_router_llm():
    """Trả về LLM dùng cho Node 1 (Router) và Node 4 (Direct Response)."""
    global _llm_groq_router
    if not USE_GROQ_FOR_ROUTER:
        return _get_gemini_llm()

    if _llm_groq_router is None:
        load_dotenv(ENV_PATH)
        api_key = os.getenv(GROQ_API_KEY_ENV_NAME)
        if not api_key:
            return _get_gemini_llm()

        try:
            _llm_groq_router = ChatGroq(
                model=GROQ_ROUTER_MODEL,
                temperature=LLM_TEMPERATURE,
                api_key=api_key
            )
        except Exception:
            return _get_gemini_llm()

    return _llm_groq_router


def _get_expansion_llm():
    """Trả về LLM dùng cho Node 3 (Expansion)."""
    global _llm_groq_expansion
    if not USE_GROQ_FOR_ROUTER:
        return _get_gemini_llm()

    if _llm_groq_expansion is None:
        load_dotenv(ENV_PATH)
        api_key = os.getenv(GROQ_API_KEY_ENV_NAME)
        if not api_key:
            return _get_gemini_llm()

        try:
            _llm_groq_expansion = ChatGroq(
                model=GROQ_EXPANSION_MODEL,
                temperature=LLM_TEMPERATURE,
                api_key=api_key
            )
        except Exception:
            return _get_gemini_llm()

    return _llm_groq_expansion


def _get_direct_response_llm():
    """Trả về LLM dùng cho Node 4 (Direct Response)."""
    return _get_router_llm()


# ============================================================
# DEFAULT INTENT (dùng khi LLM parse thất bại)
# ============================================================

_DEFAULT_ENTITIES: dict = {
    "genres": [],
    "emotions": [],
    "constraints": [],
    "similar_to": [],
    "social_context": [],
}


def _extract_text(content) -> str:
    """
    Trích xuất text từ response.content của Gemini.

    ChatGoogleGenerativeAI có thể trả về:
    - str: format cũ / text đơn giản
    - list[dict]: format mới, mỗi dict có key "text"

    Returns:
        Chuỗi văn bản đã strip.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        # Ghép tất cả các block có key 'text'
        parts = [block["text"] for block in content if isinstance(block, dict) and "text" in block]
        return "\n".join(parts).strip()
    # Fallback: ép sang str
    return str(content).strip()


# ============================================================
# NODE 1: QUERY ANALYZER (ROUTER)
# ============================================================

def analyzer_node(state: RecSysState) -> dict:
    """
    Node 1: Query Analyzer — Phân tích intent và điều hướng graph.

    Gọi Gemini để phân loại câu query thành 1 trong 3 intent:
    - "search_movie":       Query rõ ràng, đi thẳng vào Fast FAISS (Node 2).
    - "need_expansion":     Query mơ hồ/cảm xúc, cần mở rộng (Node 3).
    - "not_recommendation": Không phải yêu cầu tìm phim (Node 4).

    Nếu intent = "search_movie", set search_query = original_query.

    Args:
        state:
            Trạng thái hiện tại của graph. Cần có:
            - original_query: str

    Returns:
        Dict cập nhật vào state:
        - intent: str
        - intent_reasoning: str
        - extracted_entities: dict
        - search_query: str (chỉ set khi intent = "search_movie")
        - execution_path: thêm "analyzer"
        - timings: thời gian chạy node
    """
    start_time = time.perf_counter()

    original_query = state["original_query"]
    llm = _get_router_llm()

    # ── Gọi LLM để phân loại intent ──
    messages = [
        ("system", ROUTER_SYSTEM_PROMPT),
        ("human", ROUTER_USER_TEMPLATE.format(query=original_query)),
    ]

    try:
        response = llm.invoke(messages)
        raw_content = _extract_text(response.content)

        # Bóc JSON — Gemini đôi khi wrap trong ```json ... ```
        if raw_content.startswith("```"):
            lines = raw_content.splitlines()
            # Bỏ dòng đầu (```json) và dòng cuối (```)
            raw_content = "\n".join(lines[1:-1]).strip()

        parsed = json.loads(raw_content)
        intent = parsed.get("intent", "search_movie")
        reasoning = parsed.get("reasoning", "")
        entities = parsed.get("entities", _DEFAULT_ENTITIES)

    except (json.JSONDecodeError, Exception):
        # Fallback an toàn: nếu parse thất bại → xem như search_movie
        # để tránh user bị stuck, và log fallback vào reasoning
        intent = "search_movie"
        reasoning = "LLM parse failed — fallback to search_movie"
        entities = _DEFAULT_ENTITIES

    # Validate intent hợp lệ
    valid_intents = {"search_movie", "need_expansion", "not_recommendation"}
    if intent not in valid_intents:
        intent = "search_movie"
        reasoning = f"Unknown intent from LLM — fallback to search_movie. Original reasoning: {reasoning}"

    elapsed = time.perf_counter() - start_time

    # Nếu query rõ ràng → set search_query = original_query
    # để các node phía sau (fast_retrieval) dùng trực tiếp
    update: dict = {
        "intent": intent,
        "intent_reasoning": reasoning,
        "extracted_entities": entities,
        "execution_path": state.get("execution_path", []) + ["analyzer"],
        "timings": {
            **state.get("timings", {}),
            "analyzer": round(elapsed, 4),
        },
    }

    if intent == "search_movie":
        update["search_query"] = original_query

    return update


# ============================================================
# NODE 2: FAST FAISS RETRIEVAL (FAST PATH)
# ============================================================

def fast_retrieval_node(
    state: RecSysState,
    vector_store,
) -> dict:
    """
    Node 2: Fast FAISS Retrieval (Fast Path — Bỏ qua Cross-Encoder).

    Chạy FAISS retrieval + scoring nhẹ (semantic + popularity + rule)
    mà KHÔNG dùng Cross-Encoder. Đây là đường tắt cho các query
    đơn giản, rõ ràng.

    Args:
        state:
            Trạng thái hiện tại của graph. Cần có:
            - search_query hoặc original_query
            - top_n (optional, default FINAL_RECOMMENDATION_TOP_K)
        vector_store:
            FAISS vector store đã load.

    Returns:
        Dict cập nhật vào state:
        - ranked_movies: DataFrame đã xếp hạng (không có CE score)
        - used_cross_encoder: False
        - execution_path: thêm "fast_retrieval"
        - timings: thời gian chạy node
    """
    start_time = time.perf_counter()

    # Lấy query: ưu tiên search_query (đã set bởi Router),
    # fallback về original_query
    query = state.get("search_query") or state["original_query"]
    top_n = state.get("top_n", FINAL_RECOMMENDATION_TOP_K)

    # Bước 1: FAISS retrieval
    candidates = retrieve_movies_with_score(
        vector_store=vector_store,
        query=query,
        k=RETRIEVAL_TOP_K,
    )
    candidates_df = pd.DataFrame(candidates)

    # Bước 2: Scoring nhẹ (KHÔNG có Cross-Encoder)
    result = rank_without_ce(
        candidates_df=candidates_df,
        query=query,
    )
    result = result.head(top_n).reset_index(drop=True)

    elapsed = time.perf_counter() - start_time

    return {
        "ranked_movies": result,
        "used_cross_encoder": False,
        "execution_path": state.get("execution_path", [])
        + ["fast_retrieval"],
        "timings": {
            **state.get("timings", {}),
            "fast_retrieval": round(elapsed, 4),
        },
    }


# ============================================================
# NODE 3: QUERY EXPANSION
# ============================================================

def expansion_node(state: RecSysState) -> dict:
    """
    Node 3: Query Expansion — Viết lại query mơ hồ thành mô tả nội dung phim.

    "Dịch" ngữ cảnh người xem / cảm xúc trừu tượng thành ngôn ngữ
    mô tả nội dung phim mà FAISS và Cross-Encoder hiểu được.

    - Lần đầu (retry_count = 0): Dùng EXPANSION_SYSTEM_PROMPT.
    - Retry (retry_count > 0): Dùng EXPANSION_RETRY_PROMPT với context
      từ lần expansion trước để tạo query đa dạng hơn.

    Args:
        state:
            Trạng thái hiện tại của graph. Cần có:
            - original_query: str
            Có thể có:
            - retry_count: int (mặc định 0)
            - previous_expanded_query: str (từ lần expansion trước)

    Returns:
        Dict cập nhật vào state:
        - expanded_query: str (query mới đã mở rộng)
        - search_query: str (= expanded_query, dùng cho FAISS)
        - previous_expanded_query: str (lưu lại cho retry)
        - execution_path: thêm "expand"
        - timings: thời gian chạy node
    """
    start_time = time.perf_counter()

    original_query = state["original_query"]
    retry_count = state.get("retry_count", 0)
    previous_expanded = state.get("previous_expanded_query")
    llm = _get_expansion_llm()

    # ── Chọn prompt tùy theo lần đầu hay retry ──
    if retry_count > 0 and previous_expanded:
        # Retry: dùng EXPANSION_RETRY_PROMPT với context lần trước
        messages = [
            (
                "human",
                EXPANSION_RETRY_PROMPT.format(
                    previous_expanded_query=previous_expanded,
                    original_query=original_query,
                ),
            ),
        ]
    else:
        # Lần đầu: dùng prompt tiêu chuẩn
        messages = [
            ("system", EXPANSION_SYSTEM_PROMPT),
            ("human", EXPANSION_USER_TEMPLATE.format(query=original_query)),
        ]

    try:
        response = llm.invoke(messages)
        expanded_query = _extract_text(response.content)

        # Loại bỏ dấu ngoặc kép bao quanh (nếu LLM trả về)
        expanded_query = expanded_query.strip('"').strip("'")

    except Exception:
        # Fallback: dùng query gốc nếu expansion thất bại
        expanded_query = original_query

    elapsed = time.perf_counter() - start_time

    return {
        "expanded_query": expanded_query,
        "search_query": expanded_query,
        "previous_expanded_query": expanded_query,
        "execution_path": state.get("execution_path", []) + ["expand"],
        "timings": {
            **state.get("timings", {}),
            "expand": round(elapsed, 4),
        },
    }


# ============================================================
# NODE 4: DIRECT RESPONSE (CHITCHAT / CLARIFICATION)
# ============================================================

def direct_response_node(state: RecSysState) -> dict:
    """
    Node 4: Direct Response — Xử lý các câu không liên quan đến gợi ý phim.

    Gọi Gemini trả lời lịch sự, ngắn gọn bằng tiếng Việt.
    Không chạy FAISS, không chạy Cross-Encoder → tiết kiệm hoàn toàn
    tài nguyên. Kết thúc graph ngay sau node này (→ END).

    Args:
        state:
            Trạng thái hiện tại của graph. Cần có:
            - original_query: str

    Returns:
        Dict cập nhật vào state:
        - direct_response: str (câu trả lời bằng tiếng Việt)
        - execution_path: thêm "direct_response"
        - timings: thời gian chạy node
    """
    start_time = time.perf_counter()

    original_query = state["original_query"]
    llm = _get_direct_response_llm()

    messages = [
        ("system", DIRECT_RESPONSE_SYSTEM_PROMPT),
        ("human", DIRECT_RESPONSE_USER_TEMPLATE.format(query=original_query)),
    ]

    try:
        response = llm.invoke(messages)
        direct_response = _extract_text(response.content)
    except Exception:
        direct_response = (
            "Xin chào! Tôi là RecSysLab, hệ thống gợi ý phim AI. "
            "Bạn có thể hỏi tôi về bất kỳ thể loại phim nào, "
            "ví dụ: \"phim hành động\", \"phim hài lãng mạn\", \"phim kinh dị\"."
        )

    elapsed = time.perf_counter() - start_time

    return {
        "direct_response": direct_response,
        "execution_path": state.get("execution_path", [])
        + ["direct_response"],
        "timings": {
            **state.get("timings", {}),
            "direct_response": round(elapsed, 4),
        },
    }


# ============================================================
# NODE 5: FULL HYBRID + CROSS-ENCODER RERANKING (QUALITY PATH)
# ============================================================

def full_reranking_node(
    state: RecSysState,
    vector_store,
    cross_encoder,
) -> dict:
    """
    Node 5: Full Hybrid + CE Reranking (Quality Path).

    Chạy pipeline hoàn chỉnh: FAISS retrieval + 4-Factor Scoring
    + Cross-Encoder. Đây là đường chất lượng cao dành cho các query
    phức tạp / đã được mở rộng bởi Node 3.

    **IMPORTANT - Query Usage Strategy:**
    - FAISS Retrieval: Dùng expanded query (broad recall)
    - Cross-Encoder Reranking: Dùng original query (precision scoring)

    Rationale: Expanded query giúp FAISS retrieve candidates đa dạng,
    nhưng CE cần original query để đánh giá chính xác intent gốc của user
    (tránh semantic drift từ expansion).

    Args:
        state:
            Trạng thái hiện tại của graph. Cần có:
            - search_query: str (đã expand từ Node 3, dùng cho FAISS)
            - original_query: str (query gốc, dùng cho Cross-Encoder)
            - top_n (optional, default FINAL_RECOMMENDATION_TOP_K)
        vector_store:
            FAISS vector store đã load.
        cross_encoder:
            Cross-Encoder model instance.

    Returns:
        Dict cập nhật vào state:
        - ranked_movies: DataFrame đã xếp hạng (có cross_encoder_score)
        - used_cross_encoder: True
        - execution_path: thêm "full_reranking"
        - timings: thời gian chạy node
    """
    start_time = time.perf_counter()

    expanded_query = state["search_query"]  # For FAISS (broad recall)
    original_query = state["original_query"]  # For CE (precision)
    top_n = state.get("top_n", FINAL_RECOMMENDATION_TOP_K)

    # Bước 1: FAISS retrieval với expanded query
    # Expanded query giúp retrieve candidates đa dạng hơn
    candidates = retrieve_movies_with_score(
        vector_store=vector_store,
        query=expanded_query,
        k=RETRIEVAL_TOP_K,
    )
    candidates_df = pd.DataFrame(candidates)

    # Bước 2: Full reranking (4-factor + Cross-Encoder)
    # Cross-Encoder dùng original query để đánh giá chính xác
    # intent gốc, tránh semantic drift từ expansion
    ranked = rank_with_ce(
        candidates_df=candidates_df,
        query=original_query,
        cross_encoder=cross_encoder,
    )
    ranked = ranked.head(top_n).reset_index(drop=True)

    elapsed = time.perf_counter() - start_time

    return {
        "ranked_movies": ranked,
        "used_cross_encoder": True,
        "execution_path": state.get("execution_path", [])
        + ["full_reranking"],
        "timings": {
            **state.get("timings", {}),
            "full_reranking": round(elapsed, 4),
        },
    }


# ============================================================
# NODE 6: QUALITY GATE (ADAPTIVE EVALUATOR & SELF-CORRECTION)
# ============================================================

def quality_gate_node(state: RecSysState) -> dict:
    """
    Node 6: Quality Gate — Đánh giá chất lượng kết quả và quyết định
    tiếp tục hay retry.

    Nhận kết quả từ **cả hai đường**:
    - Fast Path (Node 2): Đánh giá bằng mean Semantic Similarity.
    - Quality Path (Node 5): Đánh giá bằng mean Cross-Encoder score.

    Nếu chất lượng thấp VÀ retry_count < MAX_RETRIES → quay lại
    Node 3 (Expansion) để retry. Nếu đã hết retry → chấp nhận
    kết quả hiện tại.

    Args:
        state:
            Trạng thái hiện tại của graph. Cần có:
            - ranked_movies: pd.DataFrame (kết quả từ Node 2 hoặc Node 5)
            - used_cross_encoder: bool (True nếu từ Node 5)
            Có thể có:
            - retry_count: int (mặc định 0)

    Returns:
        Dict cập nhật vào state:
        - mean_cross_encoder_score: float | None
        - mean_semantic_score: float
        - confidence_level: "high" | "low"
        - retry_count: int (tăng 1 nếu confidence = "low")
        - execution_path: thêm "quality_gate"
        - timings: thời gian chạy node
    """
    start_time = time.perf_counter()

    ranked = state["ranked_movies"]
    used_ce = state.get("used_cross_encoder", False)
    retry_count = state.get("retry_count", 0)

    # ── Tính mean scores ──
    mean_semantic = ranked["semantic_similarity"].mean()

    if used_ce:
        # Quality Path: đánh giá bằng CE score
        mean_ce = ranked["cross_encoder_score"].mean()
        is_good = mean_ce >= CE_CONFIDENCE_THRESHOLD
    else:
        # Fast Path: đánh giá bằng Semantic score (không có CE)
        mean_ce = None
        is_good = mean_semantic >= SEMANTIC_CONFIDENCE_THRESHOLD

    # ── Quyết định: tiếp tục hay retry ──
    if is_good or retry_count >= MAX_RETRIES:
        confidence = "high"
    else:
        confidence = "low"

    elapsed = time.perf_counter() - start_time

    return {
        "mean_cross_encoder_score": round(mean_ce, 4) if mean_ce is not None else None,
        "mean_semantic_score": round(mean_semantic, 4),
        "confidence_level": confidence,
        "retry_count": retry_count + 1 if confidence == "low" else retry_count,
        "execution_path": state.get("execution_path", []) + ["quality_gate"],
        "timings": {
            **state.get("timings", {}),
            "quality_gate": round(elapsed, 4),
        },
    }


# ============================================================
# NODE 7: EXPLANATION NODE
# ============================================================

def explanation_node(state: RecSysState) -> dict:
    """
    Node 7: LLM Explanation — Tạo giải thích cho từng phim.

    Chỉ chạy nếu include_explanation = True.
    Sử dụng original_query để explanation phản ánh đúng câu hỏi gốc.

    Args:
        state: Trạng thái hiện tại của graph.

    Returns:
        Dict cập nhật vào state:
        - explanations: list[str]
        - execution_path: thêm "explanation"
        - timings: thời gian chạy node
    """
    start_time = time.perf_counter()

    if not state.get("include_explanation", True):
        elapsed = time.perf_counter() - start_time
        return {
            "explanations": [],
            "execution_path": state.get("execution_path", []) + ["explanation"],
            "timings": {
                **state.get("timings", {}),
                "explanation": round(elapsed, 4),
            },
        }

    explanations = explain_ranked_movies(
        query=state["original_query"],
        ranked_movies=state["ranked_movies"],
    )

    elapsed = time.perf_counter() - start_time

    return {
        "explanations": explanations,
        "execution_path": state.get("execution_path", []) + ["explanation"],
        "timings": {
            **state.get("timings", {}),
            "explanation": round(elapsed, 4),
        },
    }
