from src.config import (
    FAISS_REFACTORED_INDEX_DIR,
    FINAL_RECOMMENDATION_TOP_K,
    RETRIEVAL_TOP_K,
)
from src.generation.explanation_chain import (
    explain_ranked_movies,
    format_movies_for_llm,
)
from src.ranking.hybrid_ranker import rank_movies
from src.retrieval.embeddings import load_embedding_model
from src.retrieval.vector_store import load_vector_store


TEST_QUERY = "psychological sci-fi movies"


def main() -> None:
    print("=" * 80)
    print("TEST FULL PIPELINE: RETRIEVAL → RANKING → GENERATION")
    print("=" * 80)

    print("\n[1/5] Load embedding model...")
    embedding_model = load_embedding_model()

    print("\n[2/5] Load FAISS vector store...")
    vector_store = load_vector_store(
        embedding_model=embedding_model,
        index_dir=FAISS_REFACTORED_INDEX_DIR,
    )

    print(f"\n[3/5] Rank movies for query: '{TEST_QUERY}'")

    ranked_movies = rank_movies(
        vector_store=vector_store,
        query=TEST_QUERY,
        retrieval_k=RETRIEVAL_TOP_K,
        top_n=FINAL_RECOMMENDATION_TOP_K,
    )

    print("\nTOP MOVIES AFTER RANKING:")

    print(
        ranked_movies[
            [
                "final_rank",
                "title",
                "genres",
                "cross_encoder_score",
                "final_score",
            ]
        ].to_string(index=False)
    )

    print("\n[4/5] Preview LLM context...")

    context = format_movies_for_llm(ranked_movies)

    print("\n" + "-" * 80)
    print(context)
    print("-" * 80)

    print("\n[5/5] Generate explanation...")

    explanation = explain_ranked_movies(
        query=TEST_QUERY,
        ranked_movies=ranked_movies,
    )

    print("\n" + "=" * 80)
    print("FINAL MARKDOWN RESPONSE")
    print("=" * 80)
    print(explanation)
    print("=" * 80)


if __name__ == "__main__":
    main()