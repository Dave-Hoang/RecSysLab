from pathlib import Path
from typing import Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import FAISS_INDEX_DIR
from src.data.documents import load_documents
from src.retrieval.embeddings import load_embedding_model


def build_vector_store(
    documents: Sequence[Document],
    embedding_model: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Tạo FAISS vector store từ danh sách LangChain Documents.

    Args:
        documents:
            Danh sách Document cần index.
        embedding_model:
            Model dùng để sinh embedding.

    Returns:
        FAISS vector store nằm trong bộ nhớ.

    Raises:
        ValueError:
            Nếu danh sách Document rỗng.
    """
    if not documents:
        raise ValueError(
            "Không thể build FAISS index vì danh sách Documents rỗng."
        )

    print(
        f"[*] Đang tạo embeddings và build FAISS cho "
        f"{len(documents):,} documents..."
    )

    vector_store = FAISS.from_documents(
        documents=list(documents),
        embedding=embedding_model,
    )

    print("[✓] Build FAISS vector store thành công.")

    return vector_store


def save_vector_store(
    vector_store: FAISS,
    output_dir: Path = FAISS_INDEX_DIR,
) -> None:
    """
    Lưu FAISS vector store xuống ổ đĩa.

    Args:
        vector_store:
            FAISS object cần lưu.
        output_dir:
            Thư mục output.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Đang lưu FAISS index tại: {output_dir}")

    vector_store.save_local(str(output_dir))

    index_file = output_dir / "index.faiss"
    metadata_file = output_dir / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        raise RuntimeError(
            "FAISS đã chạy save_local() nhưng không tạo đủ "
            "index.faiss và index.pkl."
        )

    print("[✓] Đã lưu FAISS index thành công.")


def load_vector_store(
    embedding_model: HuggingFaceEmbeddings,
    index_dir: Path = FAISS_INDEX_DIR,
    allow_dangerous_deserialization: bool = True,
) -> FAISS:
    """
    Load FAISS index đã được lưu trước đó.

    Chỉ bật allow_dangerous_deserialization với index do chính
    project này tạo hoặc từ nguồn đáng tin cậy.

    Args:
        embedding_model:
            Model embedding phải giống cấu hình khi build index.
        index_dir:
            Thư mục chứa index.faiss và index.pkl.
        allow_dangerous_deserialization:
            Cho phép LangChain load file pickle.

    Returns:
        FAISS vector store.

    Raises:
        FileNotFoundError:
            Nếu thiếu index.faiss hoặc index.pkl.
    """
    index_file = index_dir / "index.faiss"
    metadata_file = index_dir / "index.pkl"

    missing_files = [
        path
        for path in (index_file, metadata_file)
        if not path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(
            f"- {path}" for path in missing_files
        )

        raise FileNotFoundError(
            "Không tìm thấy đầy đủ FAISS index:\n"
            f"{missing_text}\n"
            "Hãy chạy script build_faiss_index trước."
        )

    print(f"[*] Đang load FAISS index từ: {index_dir}")

    vector_store = FAISS.load_local(
        folder_path=str(index_dir),
        embeddings=embedding_model,
        allow_dangerous_deserialization=(
            allow_dangerous_deserialization
        ),
    )

    print("[✓] Load FAISS index thành công.")

    return vector_store


def build_and_save_vector_store(
    output_dir: Path = FAISS_INDEX_DIR,
) -> FAISS:
    """
    Chạy toàn bộ pipeline build FAISS:

        processed Parquet
        → LangChain Documents
        → embedding model
        → FAISS
        → save local

    Args:
        output_dir:
            Thư mục lưu FAISS index.

    Returns:
        FAISS vector store vừa được build.
    """
    print("[1/4] Đang load LangChain Documents...")
    documents = load_documents()

    print(f"[✓] Số Documents: {len(documents):,}")

    print("[2/4] Đang load embedding model...")
    embedding_model = load_embedding_model()

    print("[3/4] Đang build vector store...")
    vector_store = build_vector_store(
        documents=documents,
        embedding_model=embedding_model,
    )

    print("[4/4] Đang lưu vector store...")
    save_vector_store(
        vector_store=vector_store,
        output_dir=output_dir,
    )

    return vector_store