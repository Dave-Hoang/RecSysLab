from __future__ import annotations

from src.production_config import load_production_settings

def test_load_production_settings() -> None:
    settings = load_production_settings()
    
    assert settings.default_mode == "quality"
    assert "quality" in settings.modes
    assert "fast" in settings.modes
    
    assert settings.retrieval.candidate_k > 0
    assert settings.retrieval.min_top_k > 0
    assert settings.retrieval.max_top_k >= settings.retrieval.min_top_k
