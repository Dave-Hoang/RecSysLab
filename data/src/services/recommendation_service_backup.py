from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd
from langchain_community.vectorstores import FAISS

from src.config import (
    FAISS_REFACTORED_INDEX_DIR,
    FINAL_RECOMMENDATION_TOP_K,
    RETRIEVAL_TOP_K,
)
from src.generation.explanation_chain import explain_ranked_movies
from src.ranking.hybrid_ranker import rank_movies
from src.retrieval.embeddings import load_embedding_model
from src.retrieval.vector_store import load_vector_store


@dataclass
class RecommendationResult:
    """
    Kết quả cuối cùng của recommendation service.

    Attributes:
        query:
            Query gốc của người dùng.
        recommendations:
            Danh sách phim đã được ranking.
        explanation:
            Markdown explanation do LLM tạo.
        timings:
            Thời gian chạy của từng giai đoạn.
    """

    query: str
    recommendations: list[dict[str, Any]]
    explanation: str
    timings: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """
        Chuyển kết quả thành dictionary để dùng cho FastAPI hoặc Streamlit.
        """
        return {
            "query": self.query,
            "recommendations": self.recommendations,
            "explanation": self.explanation,
            "timings": self.timings,
        }


class RecommendationService:
    """
    Service điều phối toàn bộ movie recommendation pipeline.

    Service chịu trách nhiệm:
    - load embedding model một lần;
    - load FAISS index một lần;
    - gọi tầng ranking;
    - gọi tầng generation;
    - trả kết quả có cấu trúc.
    """

    def __init__(
        self,
        vector_store: FAISS | None = None,
        retrieval_k: int = RETRIEVAL_TOP_K,
        top_n: int = FINAL_RECOMMENDATION_TOP_K,
    ) -> None:
        """
        Khởi tạo RecommendationService.

        Args:
            vector_store:
                FAISS vector store đã load sẵn.
                Nếu None, service sẽ tự load.
            retrieval_k:
                Số candidates lấy từ FAISS.
            top_n:
                Số phim cuối cùng trả về.
        """
        if retrieval_k <= 0:
            raise ValueError("retrieval_k phải lớn hơn 0.")

        if top_n <= 0:
            raise ValueError("top_n phải lớn hơn 0.")

        if top_n > retrieval_k:
            raise ValueError(
                "top_n không được lớn hơn retrieval_k."
            )

        self.retrieval_k = retrieval_k
        self.top_n = top_n

        if vector_store is None:
            self.vector_store = self._initialize_vector_store()
        else:
            self.vector_store = vector_store

    @staticmethod
    def _initialize_vector_store() -> FAISS:
        """
        Load embedding model và FAISS index.

        Hàm này chỉ được gọi khi service được khởi tạo,
        không chạy lại ở mỗi query.
        """
        print("[Service] Đang load embedding model...")

        embedding_model = load_embedding_model()

        print("[Service] Đang load FAISS vector store...")

        vector_store = load_vector_store(
            embedding_model=embedding_model,
            index_dir=FAISS_REFACTORED_INDEX_DIR,
        )

        print("[Service] Recommendation service đã sẵn sàng.")

        return vector_store

    @staticmethod
    def _validate_query(query: str) -> str:
        """
        Kiểm tra và chuẩn hóa query.

        Returns:
            Query đã strip khoảng trắng.

        Raises:
            TypeError:
                Nếu query không phải string.
            ValueError:
                Nếu query rỗng.
        """
        if not isinstance(query, str):
            raise TypeError("Query phải là chuỗi.")

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("Query không được để trống.")

        return cleaned_query

    @staticmethod
    def _dataframe_to_records(
        ranked_movies: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Chuyển ranked DataFrame thành list dictionary.

        Dữ liệu này phù hợp để serialize thành JSON.
        """
        if ranked_movies.empty:
            return []

        safe_frame = ranked_movies.copy()

        numeric_columns = [
            "rating_mean",
            "rating_count",
            "faiss_distance",
            "semantic_similarity",
            "popularity_score",
            "rule_score",
            "cross_encoder_score",
            "final_score",
            "rank",
            "final_rank",
            "movieId",
        ]

        for column in numeric_columns:
            if column in safe_frame.columns:
                safe_frame[column] = pd.to_numeric(
                    safe_frame[column],
                    errors="coerce",
                )

        safe_frame = safe_frame.where(
            pd.notna(safe_frame),
            None,
        )

        return safe_frame.to_dict(orient="records")

    def rank(
        self,
        query: str,
    ) -> pd.DataFrame:
        """
        Chạy retrieval và ranking, chưa gọi LLM.

        Hữu ích cho:
        - debugging;
        - evaluation;
        - benchmark;
        - API chỉ cần kết quả phim.
        """
        cleaned_query = self._validate_query(query)

        return rank_movies(
            vector_store=self.vector_store,
            query=cleaned_query,
            retrieval_k=self.retrieval_k,
            top_n=self.top_n,
        )

    def recommend(
        self,
        query: str,
        include_explanation: bool = True,
    ) -> RecommendationResult:
        """
        Chạy pipeline recommendation hoàn chỉnh.

        Pipeline:
            query
            → FAISS retrieval
            → hybrid reranking
            → Top-N movies
            → Gemini explanation

        Args:
            query:
                Query của người dùng.
            include_explanation:
                Nếu False, chỉ trả recommendations và bỏ qua LLM.

        Returns:
            RecommendationResult có cấu trúc.
        """
        cleaned_query = self._validate_query(query)

        total_start = perf_counter()

        ranking_start = perf_counter()

        ranked_movies = rank_movies(
            vector_store=self.vector_store,
            query=cleaned_query,
            retrieval_k=self.retrieval_k,
            top_n=self.top_n,
        )

        ranking_time = perf_counter() - ranking_start

        explanation = ""
        generation_time = 0.0

        if include_explanation:
            generation_start = perf_counter()

            explanation = explain_ranked_movies(
                query=cleaned_query,
                ranked_movies=ranked_movies,
            )

            generation_time = (
                perf_counter() - generation_start
            )

        total_time = perf_counter() - total_start

        recommendations = self._dataframe_to_records(
            ranked_movies
        )

        timings = {
            "ranking_seconds": round(ranking_time, 4),
            "generation_seconds": round(
                generation_time,
                4,
            ),
            "total_seconds": round(total_time, 4),
        }

        return RecommendationResult(
            query=cleaned_query,
            recommendations=recommendations,
            explanation=explanation,
            timings=timings,
        )