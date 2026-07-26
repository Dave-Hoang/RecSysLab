from __future__ import annotations

from enum import Enum


class QueryCategory(str, Enum):
    GENRE = "genre"
    EMOTION_THEME = "emotion_theme"
    MULTI_CONDITION = "multi_condition"
    SIMILAR_MOVIE = "similar_movie"
    NEGATIVE_CONSTRAINT = "negative_constraint"
    NATURAL_LANGUAGE = "natural_language"


class QueryDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class RankingConfiguration(str, Enum):
    FAISS_ONLY = "faiss_only"
    HYBRID_NO_CE = "hybrid_no_ce"
    CROSS_ENCODER_ONLY = "cross_encoder_only"
    HYBRID_WITH_CE = "hybrid_with_ce"


EVALUATION_RETRIEVAL_K = 20
EVALUATION_TOP_K = 5


HYBRID_NO_CE_WEIGHTS = {
    "semantic": 0.70,
    "popularity": 0.20,
    "rule": 0.10,
}


HYBRID_WITH_CE_WEIGHTS = {
    "cross_encoder": 0.50,
    "semantic": 0.25,
    "popularity": 0.15,
    "rule": 0.10,
}