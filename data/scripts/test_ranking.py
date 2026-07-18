import pandas as pd

from src.config import (
    FAISS_REFACTORED_INDEX_DIR,
    FINAL_RECOMMENDATION_TOP_K,
    RETRIEVAL_TOP_K,
)
from src.ranking.hybrid_ranker import rank_movies
from src.retrieval.embeddings import load_embedding_model
from src.retrieval.vector_store import load_vector_store


TEST_QUERIES = [
    "scary ghost movie",
    "sad movie about father and son",
    "romantic comedy about fake relationship",
]


DISPLAY_COLUMNS = [
    "final_rank",
    "title",
    "genres",
    "semantic_similarity",
    "popularity_score",
    "rule_score",
    "cross_encoder_score",
    "final_score",
]


def print_ranking_results(
    ranked_movies: pd.DataFrame,
) -> None:
    available_columns = [
        column
        for column in DISPLAY_COLUMNS
        if column in ranked_movies.columns
    ]

    print(
        ranked_movies[available_columns]
        .to_string(index=False)
    )


def main() -> None:
    print("[1/3] Load embedding model...")
    embedding_model = load_embedding_model()

    print("[2/3] Load FAISS vector store...")
    vector_store = load_vector_store(
        embedding_model=embedding_model,
        index_dir=FAISS_REFACTORED_INDEX_DIR,
    )

    print("[3/3] Test hybrid ranking...")

    for query in TEST_QUERIES:
        print("\n" + "=" * 100)
        print(f"QUERY: {query}")
        print("=" * 100)

        ranked_movies = rank_movies(
            vector_store=vector_store,
            query=query,
            retrieval_k=RETRIEVAL_TOP_K,
            top_n=FINAL_RECOMMENDATION_TOP_K,
        )

        print_ranking_results(ranked_movies)


if __name__ == "__main__":
    main()