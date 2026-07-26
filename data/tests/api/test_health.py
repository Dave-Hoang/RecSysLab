from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_ok(
    client: TestClient,
) -> None:
    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["service"] == "movie-recommendation-api"
    assert payload["default_mode"] == "quality"
    assert payload["available_modes"] == [
        "fast",
        "quality",
    ]
    assert payload["vector_store_loaded"] is True