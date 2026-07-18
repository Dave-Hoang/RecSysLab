import pandas as pd

from src.config import (
    FAISS_REFACTORED_INDEX_DIR,
    RETRIEVAL_RESULTS_PATH,
    RETRIEVAL_TOP_K,
)
from src.retrieval.embeddings import load_embedding_model
from src.retrieval.retriever import (
    build_retriever,
    retrieve_movies,
    retrieve_movies_with_score,
)
from src.retrieval.vector_store import load_vector_store


TEST_QUERIES = [
    "movies like interstellar with space and philosophy",
    "romantic comedy about fake relationship",
    "sad movie about father and son",
]


def print_top_results(
    candidates: list[dict],
    top_n: int = 5,
) -> None:
    """
    In Top-N candidates ra terminal.
    """
    print_columns = [
        "rank",
        "title",
        "genres",
        "rating_mean",
        "rating_count",
        "faiss_distance",
    ]

    results_df = pd.DataFrame(candidates)

    available_columns = [
        column
        for column in print_columns
        if column in results_df.columns
    ]

    print(
        results_df[available_columns]
        .head(top_n)
        .to_string(index=False)
    )


def main() -> None:
    print("[1/4] Load embedding model...")
    embedding_model = load_embedding_model()

    print("[2/4] Load refactored FAISS index...")
    vector_store = load_vector_store(
        embedding_model=embedding_model,
        index_dir=FAISS_REFACTORED_INDEX_DIR,
    )

    print("[3/4] Build LangChain Retriever...")
    retriever = build_retriever(
        vector_store=vector_store,
        k=RETRIEVAL_TOP_K,
    )

    all_results: list[dict] = []

    print("[4/4] Test retrieval...")

    for query in TEST_QUERIES:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        documents_only = retrieve_movies(
            retriever=retriever,
            query=query,
        )

        print(
            f"Retriever.invoke() trả về "
            f"{len(documents_only)} candidates."
        )

        candidates = retrieve_movies_with_score(
            vector_store=vector_store,
            query=query,
            k=RETRIEVAL_TOP_K,
        )

        print("\nTOP 5:")
        print_top_results(
            candidates=candidates,
            top_n=5,
        )

        for candidate in candidates:
            result_row = {
                "query": query,
                **candidate,
            }
            all_results.append(result_row)

    results_df = pd.DataFrame(all_results)

    RETRIEVAL_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RETRIEVAL_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\n[✓] Đã lưu kết quả retrieval tại: "
        f"{RETRIEVAL_RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()