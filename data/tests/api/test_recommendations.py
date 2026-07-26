from __future__ import annotations

from fastapi.testclient import TestClient


def test_quality_mode_uses_hybrid_with_ce(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommendations",
        json={
            "query": "psychological sci-fi movies",
            "mode": "quality",
            "top_k": 5,
            "include_explanation": False,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["mode"] == "quality"
    assert payload["configuration"] == "hybrid_with_ce"
    assert payload["top_k"] == 5
    assert len(payload["recommendations"]) == 5
    assert payload["explanation"] == ""
    assert payload["timings"]["generation_seconds"] == 0.0


def test_fast_mode_uses_hybrid_no_ce(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommendations",
        json={
            "query": "romantic comedy movies",
            "mode": "fast",
            "top_k": 3,
            "include_explanation": False,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["mode"] == "fast"
    assert payload["configuration"] == "hybrid_no_ce"
    assert payload["top_k"] == 3
    assert len(payload["recommendations"]) == 3


def test_default_mode_is_quality(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommendations",
        json={
            "query": "sad movie about father and son",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["mode"] == "quality"
    assert payload["configuration"] == "hybrid_with_ce"
    assert payload["top_k"] == 5


def test_explanation_can_be_enabled(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommendations",
        json={
            "query": "psychological sci-fi movies",
            "mode": "quality",
            "include_explanation": True,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["explanation"] == "Test explanation."
    assert payload["timings"]["generation_seconds"] > 0


def test_unknown_mode_returns_400(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommendations",
        json={
            "query": "science fiction movies",
            "mode": "unknown",
        },
    )

    assert response.status_code == 400
    assert "Mode không hợp lệ" in response.json()["detail"]


def test_top_k_above_limit_returns_400(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommendations",
        json={
            "query": "science fiction movies",
            "mode": "quality",
            "top_k": 100,
        },
    )

    assert response.status_code == 400
    assert "top_k phải nằm trong khoảng" in response.json()["detail"]


def test_blank_query_returns_422(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommendations",
        json={
            "query": " ",
            "mode": "quality",
        },
    )

    assert response.status_code == 422


def test_unknown_request_field_returns_422(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommendations",
        json={
            "query": "science fiction movies",
            "unknown_field": True,
        },
    )

    assert response.status_code == 422