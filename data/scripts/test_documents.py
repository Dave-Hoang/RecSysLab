from src.data.documents import load_documents


def main() -> None:
    documents = load_documents()

    print(f"Số lượng Documents: {len(documents):,}")

    if not documents:
        raise RuntimeError("Không tạo được Document nào.")

    print("\nSample Document:")
    print(documents[0])


if __name__ == "__main__":
    main()

    