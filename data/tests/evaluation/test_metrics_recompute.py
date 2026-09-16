import pytest
import pandas as pd
import numpy as np

from src.evaluation.metrics import ndcg_at_k
from src.evaluation.label_merger import merge_predictions_with_labels

def test_missing_label_not_zero_strict_mode():
    """Test strict mode returns NaN if there are missing labels in Top-K"""
    # ranked relevances with missing label (pd.NA)
    ranked_relevances = [2, pd.NA, 1, 0, 2]
    ideal_relevances = [2, 2, 2, 1, 1]
    
    val = ndcg_at_k(ranked_relevances, ideal_relevances, k=5, strict_mode=True)
    assert pd.isna(val)
    
def test_missing_label_treated_as_zero_non_strict():
    """Test non-strict mode treats missing labels as 0"""
    ranked_relevances = [2, pd.NA, 1, 0, 2]
    ideal_relevances = [2, 2, 2, 1, 1]
    
    val = ndcg_at_k(ranked_relevances, ideal_relevances, k=5, strict_mode=False)
    # The missing label is treated as 0 in non-strict mode
    # dcg = (2^2-1)/1 + 0/log2(3) + (2^1-1)/log2(4) + 0 + (2^2-1)/log2(6)
    # dcg = 3 + 0 + 0.5 + 0 + 3/2.585 = 3.5 + 1.16 = 4.66
    assert not pd.isna(val)
    assert val > 0

def test_left_join_preserves_rank_and_no_compaction():
    """Test that left join preserves rank and doesn't compact"""
    predictions = pd.DataFrame({
        "query_id": ["q1", "q1", "q1", "q1", "q1"],
        "query": ["test", "test", "test", "test", "test"],
        "category": ["cat", "cat", "cat", "cat", "cat"],
        "difficulty": ["easy", "easy", "easy", "easy", "easy"],
        "configuration": ["hybrid_with_ce"] * 5,
        "rank": [1, 2, 3, 4, 5],
        "movieId": [101, 102, 103, 104, 105],
        "title": ["m1", "m2", "m3", "m4", "m5"]
    })
    
    labels = pd.DataFrame({
        "query_id": ["q1", "q1"],
        "movieId": [101, 104],
        "relevance": [2, 1],
        "notes": ["", ""]
    })
    
    from unittest.mock import patch
    with patch('src.evaluation.label_merger.validate_predictions'), \
         patch('src.evaluation.label_merger.validate_labels'), \
         patch('src.evaluation.label_merger.validate_label_coverage'):
        merged = merge_predictions_with_labels(predictions, labels)
    
    assert len(merged) == 5
    assert merged["rank"].tolist() == [1, 2, 3, 4, 5]
    assert pd.isna(merged.loc[1, "relevance"])
    assert pd.isna(merged.loc[2, "relevance"])
    assert merged.loc[3, "relevance"] == 1
    
def test_no_cartesian_product_on_duplicates():
    """Test that duplicates do not create cartesian product"""
    predictions = pd.DataFrame({
        "query_id": ["q1"],
        "query": ["test"],
        "category": ["cat"],
        "difficulty": ["easy"],
        "configuration": ["hybrid_with_ce"],
        "rank": [1],
        "movieId": [101],
        "title": ["m1"]
    })
    
    labels = pd.DataFrame({
        "query_id": ["q1", "q1"],
        "movieId": [101, 101],
        "relevance": [2, 1],
        "notes": ["", ""]
    })
    
    from src.evaluation.label_merger import validate_labels
    with pytest.raises(ValueError, match="labels.csv có cặp query_id \\+ movieId bị trùng"):
        validate_labels(labels)

if __name__ == '__main__':
    pytest.main(["-v", __file__])
