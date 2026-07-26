from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import (
    FAISS_REFACTORED_INDEX_DIR,
    FINAL_RECOMMENDATION_TOP_K,
    RETRIEVAL_TOP_K,
)
from src.evaluation.ranking_configurations import (
    rank_with_configuration,
)
from src.generation.explanation_chain import (
    explain_ranked_movies,
)
from src.retrieval.embeddings import load_embedding_model
from src.retrieval.retriever import retrieve_movies
from src.retrieval.vector_store import load_vector_store


SUPPORTED_CONFIGURATIONS = {
    "faiss_only",
    "hybrid_no_ce",
    "cross_encoder_only",
    "hybrid_with_ce",
}


@dataclass(frozen=True)
class RecommendationResult:
    """Kết quả cuối cùng của recommendation pipeline."""

    query: str
    configuration: str
    recommendations: list[dict[str, Any]]
    timings: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Chuyển kết quả thành dictionary JSON-friendly."""

        return {
            "query": self.query,
            "configuration": self.configuration,
            "recommendations": self.recommendations,
            "timings": self.timings,
        }


class RecommendationService:
    """
    Điều phối retrieval, ranking và explanation.

    Embedding model, FAISS index và retriever được khởi tạo
    một lần khi service bắt đầu, không load lại ở mỗi request.
    """

    def __init__(
        self,
        vector_store: FAISS | None = None,
        retrieval_k: int = RETRIEVAL_TOP_K,
        top_n: int = FINAL_RECOMMENDATION_TOP_K,
    ) -> None:
        self.retrieval_k = self._validate_positive_integer(
            value=retrieval_k,
            field_name="retrieval_k",
        )
        self.top_n = self._validate_positive_integer(
            value=top_n,
            field_name="top_n",
        )

        if self.top_n > self.retrieval_k:
            raise ValueError(
                "top_n không được lớn hơn retrieval_k. "
                f"Nhận được top_n={self.top_n}, "
                f"retrieval_k={self.retrieval_k}."
            )

        self.vector_store = (
            vector_store
            if vector_store is not None
            else self._initialize_vector_store()
        )

        self.retriever = self._create_retriever(
            vector_store=self.vector_store,
            retrieval_k=self.retrieval_k,
        )

    @staticmethod
    def _validate_positive_integer(
        value: int,
        field_name: str,
    ) -> int:
        """Kiểm tra một giá trị phải là số nguyên dương."""

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{field_name} phải là số nguyên."
            )

        if value <= 0:
            raise ValueError(
                f"{field_name} phải lớn hơn 0."
            )

        return value

    @staticmethod
    def _initialize_vector_store() -> FAISS:
        """Load embedding model và FAISS index một lần."""

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
    def _create_retriever(
        vector_store: FAISS,
        retrieval_k: int,
    ) -> VectorStoreRetriever:
        """Tạo retriever từ FAISS vector store."""

        return vector_store.as_retriever(
            search_kwargs={
                "k": retrieval_k,
            }
        )

    @staticmethod
    def _validate_query(query: str) -> str:
        """Kiểm tra và chuẩn hóa query."""

        if not isinstance(query, str):
            raise TypeError("Query phải là chuỗi.")

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "Query không được để trống."
            )

        if len(cleaned_query) > 500:
            raise ValueError(
                "Query không được dài quá 500 ký tự."
            )

        return cleaned_query

    def _resolve_top_n(
        self,
        top_n: int | None,
    ) -> int:
        """Kiểm tra số lượng kết quả cuối cùng."""

        selected_top_n = (
            self.top_n
            if top_n is None
            else top_n
        )

        selected_top_n = self._validate_positive_integer(
            value=selected_top_n,
            field_name="top_n",
        )

        if selected_top_n > self.retrieval_k:
            raise ValueError(
                "top_n không được lớn hơn retrieval_k "
                f"({self.retrieval_k}). "
                f"Giá trị nhận được: {selected_top_n}."
            )

        return selected_top_n

    @staticmethod
    def _validate_configuration(
        configuration: str,
    ) -> str:
        """Kiểm tra ranking configuration."""

        if not isinstance(configuration, str):
            raise TypeError(
                "configuration phải là chuỗi."
            )

        cleaned_configuration = configuration.strip()

        if cleaned_configuration not in SUPPORTED_CONFIGURATIONS:
            allowed_configurations = ", ".join(
                sorted(SUPPORTED_CONFIGURATIONS)
            )

            raise ValueError(
                "Ranking configuration không hợp lệ: "
                f"{cleaned_configuration!r}. "
                "Các configuration được hỗ trợ: "
                f"{allowed_configurations}."
            )

        return cleaned_configuration

    @staticmethod
    def _normalize_ranked_columns(
        ranked_movies: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Chuẩn hóa tên cột giữa evaluation và service output.

        Evaluation sử dụng:
            result_rank
            evaluation_score

        Service hoặc explanation chain có thể sử dụng:
            final_rank
            final_score
        """

        normalized = ranked_movies.copy()

        if (
            "result_rank" in normalized.columns
            and "final_rank" not in normalized.columns
        ):
            normalized["final_rank"] = normalized[
                "result_rank"
            ]

        if (
            "evaluation_score" in normalized.columns
            and "final_score" not in normalized.columns
        ):
            normalized["final_score"] = normalized[
                "evaluation_score"
            ]

        return normalized

    @staticmethod
    def _dataframe_to_records(
        ranked_movies: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Chuyển DataFrame thành dữ liệu JSON-friendly."""

        if ranked_movies.empty:
            return []

        safe_frame = ranked_movies.copy()

        numeric_columns = [
            "movieId",
            "rank",
            "retrieval_rank",
            "result_rank",
            "final_rank",
            "rating_mean",
            "rating_count",
            "faiss_distance",
            "semantic_similarity",
            "popularity_score",
            "rule_score",
            "cross_encoder_score",
            "evaluation_score",
            "final_score",
        ]

        for column in numeric_columns:
            if column in safe_frame.columns:
                safe_frame[column] = pd.to_numeric(
                    safe_frame[column],
                    errors="coerce",
                )

        # Thay NaN và pandas.NA bằng None để FastAPI có thể
        # serialize thành JSON null.
        safe_frame = safe_frame.astype(object).where(
            pd.notna(safe_frame),
            None,
        )

        return safe_frame.to_dict(
            orient="records",
        )

    def _retrieve_candidates(
        self,
        query: str,
    ) -> pd.DataFrame:
        """Lấy candidate từ retriever và chuyển thành DataFrame."""

        candidates = retrieve_movies(
            retriever=self.retriever,
            query=query,
        )

        if not isinstance(candidates, list):
            raise TypeError(
                "retrieve_movies() phải trả về list[dict]. "
                f"Kiểu thực tế: {type(candidates).__name__}."
            )

        candidates_dataframe = pd.DataFrame(candidates)

        if candidates_dataframe.empty:
            raise RuntimeError(
                "Retriever không trả về candidate nào."
            )

        if "title" not in candidates_dataframe.columns:
            raise RuntimeError(
                "Candidate retrieval thiếu cột bắt buộc: title."
            )

        return candidates_dataframe

    def rank(
        self,
        query: str,
        configuration: str = "hybrid_with_ce",
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """
        Chạy retrieval và ranking nhưng chưa gọi LLM.

        Luồng xử lý:
            query
            → FAISS retrieval
            → ranking theo configuration
            → Top-N
        """

        cleaned_query = self._validate_query(query)

        selected_configuration = (
            self._validate_configuration(configuration)
        )

        selected_top_n = self._resolve_top_n(top_n)

        candidates_dataframe = self._retrieve_candidates(
            query=cleaned_query,
        )

        ranked_movies = rank_with_configuration(
            query=cleaned_query,
            candidates=candidates_dataframe.copy(),
            configuration=selected_configuration,
            top_k=selected_top_n,
        )

        if not isinstance(ranked_movies, pd.DataFrame):
            raise TypeError(
                "rank_with_configuration() phải trả về "
                "pandas.DataFrame."
            )

        if ranked_movies.empty:
            raise RuntimeError(
                "Ranking pipeline không trả về kết quả."
            )

        if len(ranked_movies) > selected_top_n:
            ranked_movies = ranked_movies.head(
                selected_top_n
            )

        return self._normalize_ranked_columns(
            ranked_movies
        )

    def recommend(
        self,
        query: str,
        include_explanation: bool = True,
        configuration: str = "hybrid_with_ce",
        top_n: int | None = None,
    ) -> RecommendationResult:
        """
        Chạy recommendation pipeline hoàn chỉnh.

        Luồng xử lý:
            query
            → FAISS retrieval
            → ranking theo configuration
            → Top-N
            → optional Gemini explanation
        """

        cleaned_query = self._validate_query(query)

        selected_configuration = (
            self._validate_configuration(configuration)
        )

        selected_top_n = self._resolve_top_n(top_n)

        if not isinstance(include_explanation, bool):
            raise TypeError(
                "include_explanation phải là boolean."
            )

        total_start = perf_counter()
        ranking_start = perf_counter()

        ranked_movies = self.rank(
            query=cleaned_query,
            configuration=selected_configuration,
            top_n=selected_top_n,
        )

        ranking_seconds = (
            perf_counter() - ranking_start
        )

        explanation = ""
        generation_seconds = 0.0

        if include_explanation:
            generation_start = perf_counter()

            generated_explanations = explain_ranked_movies(
                query=cleaned_query,
                ranked_movies=ranked_movies,
            )

            ranked_movies = ranked_movies.copy()

            ranked_movies["explanation"] = generated_explanations

            generation_seconds = (
                perf_counter() - generation_start
            )

        total_seconds = (
            perf_counter() - total_start
        )

        recommendations = self._dataframe_to_records(
            ranked_movies
        )

        timings = {
            "ranking_seconds": round(
                ranking_seconds,
                4,
            ),
            "generation_seconds": round(
                generation_seconds,
                4,
            ),
            "total_seconds": round(
                total_seconds,
                4,
            ),
        }

        return RecommendationResult(
            query=cleaned_query,
            configuration=selected_configuration,
            recommendations=recommendations,
            timings=timings,
        )