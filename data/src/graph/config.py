"""
Graph-specific configuration — Thresholds và constants cho LangGraph nodes.

Các giá trị này tách riêng khỏi src/config.py (pipeline cũ)
để dễ tuning độc lập cho Agentic pipeline.
"""

# ============================================================
# QUALITY GATE THRESHOLDS
# ============================================================

# Threshold cho Quality Path (Node 5 → Node 6):
# Nếu mean Cross-Encoder score < CE_CONFIDENCE_THRESHOLD → retry.
# Giá trị 0.15 dựa trên case study Rush Hour (CE = 0.023 quá thấp).
CE_CONFIDENCE_THRESHOLD: float = 0.15

# Threshold cho Fast Path (Node 2 → Node 6):
# Nếu mean Semantic Similarity < SEMANTIC_CONFIDENCE_THRESHOLD → escalate.
SEMANTIC_CONFIDENCE_THRESHOLD: float = 0.70


# ============================================================
# RETRY / SELF-CORRECTION
# ============================================================

# Số lần retry tối đa để tránh vòng lặp vô hạn.
# Sau MAX_RETRIES lần, chấp nhận kết quả hiện tại.
MAX_RETRIES: int = 1


# ============================================================
# RETRIEVAL
# ============================================================

# Import từ config chính để giữ nhất quán
from src.config import (
    FINAL_RECOMMENDATION_TOP_K,
    RETRIEVAL_TOP_K,
)
