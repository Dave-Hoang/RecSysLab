from src.config import FAISS_REFACTORED_INDEX_DIR
from src.retrieval.vector_store import (
    build_and_save_vector_store,
)


def main() -> None:
    print("=" * 60)
    print("BUILD REFACTORED FAISS MOVIE INDEX")
    print("=" * 60)

    build_and_save_vector_store(
        output_dir=FAISS_REFACTORED_INDEX_DIR,
    )

    print("\n" + "=" * 60)
    print("FAISS INDEX BUILD COMPLETED")
    print(f"Output: {FAISS_REFACTORED_INDEX_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()