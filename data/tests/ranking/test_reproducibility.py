import os
import sys
import pytest
from pathlib import Path

# Đặt TEST_MODE=true trước khi import bất kỳ module nào của project
os.environ["TEST_MODE"] = "true"

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import TEST_MODE
from src.retrieval.vector_store import load_vector_store
from src.retrieval.embeddings import load_embedding_model
from src.ranking.cross_encoder import load_cross_encoder
from src.graph.graph import build_graph
from src.graph.nodes import _expansion_cache

# Cảnh báo:
# Dù đã bật TEST_MODE=True để ép LLM temperature=0 và cache kết quả expansion,
# tính không ổn định hiếm gặp vẫn có thể xảy ra do:
# - Hạ tầng backend LLM (Groq/Gemini API) đôi khi trả kết quả không deterministic 
#   ngay cả khi temperature=0 (flaky API).
# - Sai số floating point trên GPU khi chạy CrossEncoder / Embeddings.
# Do đó, regression test này chạy từ 3-5 lần để kiểm chứng tính ổn định thực tế.
# Đừng giảm số vòng test xuống 2 để bắt được các biến số này!

@pytest.fixture(scope="module")
def setup_models():
    """Load models once for the test module."""
    if not TEST_MODE:
        pytest.skip("Test này chỉ chạy khi TEST_MODE = True.")
        
    print("Loading embedding model...")
    embedding_model = load_embedding_model()
    
    print("Loading vector store...")
    vector_store = load_vector_store(embedding_model)
    
    print("Loading cross encoder...")
    cross_encoder = load_cross_encoder()
    
    graph_app = build_graph(vector_store, cross_encoder)
    
    return graph_app

def test_end_to_end_reproducibility(setup_models):
    """
    Test 5 lần chạy liên tiếp của LangGraph cho cùng một query.
    Đảm bảo kết quả (đặc biệt là Cross-Encoder Score) không đổi.
    """
    graph_app = setup_models
    query = "I want an emotional movie that will stay with me for days"
    
    # Xóa cache expansion trước khi chạy test để mô phỏng lần chạy thực tế đầu tiên
    _expansion_cache.clear()
    
    previous_ce_scores = None
    previous_movie_ids = None
    
    num_runs = 5
    for i in range(num_runs):
        result_state = graph_app.invoke({"original_query": query})
        ranked_movies = result_state.get("ranked_movies")
        
        assert ranked_movies is not None and not ranked_movies.empty, "Ranked movies should not be empty"
        
        movie_ids = list(ranked_movies["movieId"])
        ce_scores = list(ranked_movies["cross_encoder_score"])
        
        if previous_movie_ids is None:
            previous_movie_ids = movie_ids
            previous_ce_scores = ce_scores
        else:
            assert movie_ids == previous_movie_ids, f"Run {i+1}: Movie IDs changed. Expected {previous_movie_ids}, got {movie_ids}"
            # So sánh điểm số float với sai số chấp nhận được (1e-5)
            for j, (score1, score2) in enumerate(zip(previous_ce_scores, ce_scores)):
                assert abs(score1 - score2) < 1e-5, f"Run {i+1}: Cross Encoder score changed at rank {j+1}. Expected {score1}, got {score2}"

    # Kiểm tra ngữ nghĩa của query mở rộng (để chống hallucination)
    first_state_history = graph_app.invoke({"original_query": query})
    expanded = first_state_history.get("search_query", "").lower()
    
    print(f"\nExpanded query used in test: {expanded}")
    
    # Semantic assertions: Should contain emotional/dramatic themes
    emotional_keywords = ["emotional", "drama", "poignant", "cathartic", "lingering", "loss", "grief", "growth", "heartwarming", "moving", "touching"]
    assert any(kw in expanded for kw in emotional_keywords), f"Expanded query '{expanded}' lacks emotional/dramatic semantic themes."
    
    # Forbidden words (should not hallucinate technical details or unmentioned genres)
    forbidden_words = ["musical", "music", "score", "soundtrack", "cinematography", "visuals", "biopic", "historical", "war", "sci-fi"]
    found = [word for word in forbidden_words if word in expanded]
    assert not found, f"Expanded query '{expanded}' contains forbidden hallucinated words: {found}"
