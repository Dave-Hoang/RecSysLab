from __future__ import annotations

from time import perf_counter

import pandas as pd

from src.config import (
    EVALUATION_PREDICTIONS_PATH,
    FAISS_REFACTORED_INDEX_DIR,
)
from src.evaluation.constants import (
    EVALUATION_RETRIEVAL_K,
    EVALUATION_TOP_K,
    RankingConfiguration,
)
from src.evaluation.query_loader import (
    load_evaluation_queries,
)
from src.evaluation.ranking_configurations import (
    rank_with_configuration,
)
from src.retrieval.embeddings import load_embedding_model
from src.retrieval.retriever import retrieve_movies
from src.retrieval.vector_store import load_vector_store


def _build_prediction_rows(
    evaluation_query,
    configuration: RankingConfiguration,
    ranked_movies: pd.DataFrame,
) -> list[dict]:
    rows: list[dict] = []

    for rank_index, movie in ranked_movies.iterrows():
        rows.append(
            {
                "query_id": evaluation_query.query_id,
                "query": evaluation_query.query,
                "category": (
                    evaluation_query.category.value
                ),
                "difficulty": (
                    evaluation_query.difficulty.value
                ),
                "configuration": configuration.value,
                "rank": rank_index + 1,
                "movieId": int(movie["movieId"]),
                "title": movie["title"],
                "genres": movie["genres"],
                "faiss_distance": movie.get(
                    "faiss_distance"
                ),
                "semantic_similarity": movie.get(
                    "semantic_similarity"
                ),
                "popularity_score": movie.get(
                    "popularity_score"
                ),
                "rule_score": movie.get("rule_score"),
                "cross_encoder_score": movie.get(
                    "cross_encoder_score"
                ),
                "evaluation_score": movie[
                    "evaluation_score"
                ],
            }
        )

    return rows


def main() -> None:
    queries = load_evaluation_queries()

    print(
        f"[✓] Đã load {len(queries)} evaluation queries."
    )

    print("[*] Đang load embedding model...")
    embedding_model = load_embedding_model()

    print("[*] Đang load FAISS vector store...")

    vector_store = load_vector_store(
        embedding_model=embedding_model,
        index_dir=FAISS_REFACTORED_INDEX_DIR,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={
        "k": EVALUATION_RETRIEVAL_K,
        }
    )   

    all_rows: list[dict] = []

    total_start = perf_counter()

    for query_number, evaluation_query in enumerate(
        queries,
        start=1,
    ):
        print(
            f"\n[{query_number}/{len(queries)}] "
            f"{evaluation_query.query_id}: "
            f"{evaluation_query.query}"
        )

        candidate_records = retrieve_movies(
            retriever=retriever,
            query=evaluation_query.query,
        )

        candidates = pd.DataFrame(candidate_records)

        if candidates.empty:
            print("  [!] Candidate DataFrame rỗng.")
            continue

        print(
            f"  [✓] Retrieved {len(candidates)} candidates."
        )

        for configuration in RankingConfiguration:
            config_start = perf_counter()

            ranked_movies = rank_with_configuration(
                query=evaluation_query.query,
                candidates=candidates.copy(),
                configuration=configuration,
                top_k=EVALUATION_TOP_K,
            )

            elapsed = perf_counter() - config_start

            rows = _build_prediction_rows(
                evaluation_query=evaluation_query,
                configuration=configuration,
                ranked_movies=ranked_movies,
            )

            all_rows.extend(rows)

            print(
                f"  [✓] {configuration.value}: "
                f"{len(ranked_movies)} phim "
                f"({elapsed:.3f}s)"
            )

    predictions = pd.DataFrame(all_rows)

    EVALUATION_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        EVALUATION_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    total_elapsed = perf_counter() - total_start

    print("\n" + "=" * 80)
    print(
        f"[✓] Đã tạo {len(predictions)} prediction rows."
    )
    print(
        f"[✓] File: {EVALUATION_PREDICTIONS_PATH}"
    )
    print(f"[✓] Tổng thời gian: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()