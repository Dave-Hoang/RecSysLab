import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ranking.hybrid_ranker import _min_max_normalize, compute_cross_encoder_score
from src.ranking.cross_encoder import predict_relevance_scores, get_cross_encoder
from src.config import CROSS_ENCODER_FALLBACK_MAX_SCORE

def test_min_max_normalize_equal_values():
    """
    Test 3: input toàn giá trị giống nhau → output không chứa NaN, trả về 0.5 đồng loạt.
    """
    values = pd.Series([2.0, 2.0, 2.0])
    normalized = _min_max_normalize(values)
    
    assert not normalized.isna().any(), "Output contains NaN"
    assert (normalized == 0.5).all(), "Values should all be 0.5 when range is 0"

@patch('src.ranking.hybrid_ranker.predict_relevance_scores')
def test_cross_encoder_normalization_normal(mock_predict):
    """
    Test 1: raw logits [-3, -5, -7] → điểm cao nhất = 1.0, thấp nhất = 0.0, thứ tự giữ nguyên.
    """
    mock_predict.return_value = np.array([-3.0, -5.0, -7.0])
    
    candidates = pd.DataFrame({
        "page_content": ["doc1", "doc2", "doc3"]
    })
    
    result = compute_cross_encoder_score(candidates, query="test")
    
    scores = result["cross_encoder_score"].tolist()
    
    assert scores[0] == 1.0, f"Expected 1.0, got {scores[0]}"
    assert scores[1] == 0.5, f"Expected 0.5, got {scores[1]}"
    assert scores[2] == 0.0, f"Expected 0.0, got {scores[2]}"

@patch('src.ranking.hybrid_ranker.predict_relevance_scores')
def test_cross_encoder_normalization_fallback(mock_predict):
    """
    Test 2: raw logits dưới floor ([-9, -10, -11]) → toàn bộ output <= CROSS_ENCODER_FALLBACK_MAX_SCORE (0.3).
    """
    mock_predict.return_value = np.array([-9.0, -10.0, -11.0])
    
    candidates = pd.DataFrame({
        "page_content": ["doc1", "doc2", "doc3"]
    })
    
    result = compute_cross_encoder_score(candidates, query="test")
    scores = result["cross_encoder_score"].tolist()
    
    # rank-based fallback for [-9, -10, -11]:
    # -9 is rank 3 (max) -> 1.0 * 0.3 = 0.3
    # -10 is rank 2 -> 0.5 * 0.3 = 0.15
    # -11 is rank 1 (min) -> 0.0 * 0.3 = 0.0
    
    assert all(s <= CROSS_ENCODER_FALLBACK_MAX_SCORE for s in scores), f"Scores exceed fallback max score: {scores}"
    assert scores[0] == 0.3, f"Highest score should be 0.3, got {scores[0]}"
    assert scores[2] == 0.0, f"Lowest score should be 0.0, got {scores[2]}"

@pytest.mark.slow
def test_predict_relevance_scores_returns_raw_logit():
    """
    Test 4: Xác nhận output của predict_relevance_scores chứa ít nhất một giá trị nằm ngoài khoảng [0, 1].
    (Chống tái xuất hiện sigmoid).
    """
    ce = get_cross_encoder()
    
    # Dùng cặp query/doc không liên quan để chắc chắn lấy ra logit âm.
    scores = predict_relevance_scores(
        query="romantic comedy",
        documents=["Title: Saw. Genres: Horror, Thriller"],
        cross_encoder=ce
    )
    
    assert len(scores) == 1
    # Nếu bị dính sigmoid, điểm luôn nằm trong (0, 1)
    # Nếu là raw logit, điểm cho cặp này sẽ rất âm (ví dụ: < -5.0)
    assert scores[0] < 0.0 or scores[0] > 1.0, f"Score {scores[0]} lies in [0, 1], which indicates Sigmoid might be active!"
