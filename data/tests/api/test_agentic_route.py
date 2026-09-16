from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from unittest.mock import patch
import pandas as pd

def test_agentic_route_fast_mode(client: TestClient) -> None:
    mock_result = {
        "ranked_movies": pd.DataFrame({
            "movieId": [1, 2, 3],
            "title": ["Movie 1", "Movie 2", "Movie 3"],
            "genres": ["Action", "Action", "Action"],
            "rank": [1, 2, 3],
            "retrieval_rank": [1, 2, 3],
            "result_rank": [1, 2, 3],
            "final_rank": [1, 2, 3],
            "rating_mean": [4.0, 4.0, 4.0],
            "rating_count": [1000, 1000, 1000],
            "faiss_squared_l2_distance": [0.1, 0.1, 0.1],
            "cosine_similarity_raw": [0.9, 0.9, 0.9],
            "semantic_score_relative": [0.9, 0.9, 0.9],
            "popularity_score": [0.5, 0.5, 0.5],
            "rule_score": [0.1, 0.1, 0.1],
            "cross_encoder_score": [0.8, 0.8, 0.8],
            "evaluation_score": [0.8, 0.8, 0.8],
            "final_score": [0.8, 0.8, 0.8]
        })
    }
    from unittest.mock import Mock
    mock_graph = Mock()
    mock_graph.invoke.return_value = mock_result
    
    with patch("src.api.routes.agentic.build_graph", return_value=mock_graph):
        response = client.post(
        "/recommendations/agentic",
        json={
            "query": "romantic comedy movies",
            "top_k": 3,
            "include_explanation": False
        },
    )
    
    assert response.status_code == 200
    payload = response.json()
    assert payload["top_k"] == 3
    assert len(payload["recommendations"]) == 3
    
    # Check that explanation is not populated
    for rec in payload["recommendations"]:
        assert rec.get("explanation") in (None, "")
