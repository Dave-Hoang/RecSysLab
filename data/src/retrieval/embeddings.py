import torch
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    EMBEDDING_MODEL_NAME,
    NORMALIZE_EMBEDDINGS,
)


def get_device() -> str:
    """
    Xác định phần cứng dùng để chạy embedding model.

    Returns:
        'cuda' nếu GPU CUDA khả dụng, ngược lại là 'cpu'.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
    normalize_embeddings: bool = NORMALIZE_EMBEDDINGS,
    device: str | None = None,
) -> HuggingFaceEmbeddings:
    """
    Khởi tạo HuggingFace embedding model.

    Model này phải được sử dụng thống nhất cho cả:
    - build FAISS index;
    - load FAISS index;
    - encode user query.

    Args:
        model_name:
            Tên model trên Hugging Face.
        normalize_embeddings:
            Chuẩn hóa embedding.
        device:
            'cuda', 'cpu', hoặc None để tự động phát hiện.

    Returns:
        HuggingFaceEmbeddings đã được khởi tạo.
    """
    selected_device = device or get_device()

    print(f"[*] Embedding model: {model_name}")
    print(f"[*] Device: {selected_device.upper()}")
    print(f"[*] Normalize embeddings: {normalize_embeddings}")

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": selected_device,
        },
        encode_kwargs={
            "normalize_embeddings": normalize_embeddings,
        },
    )

    return embedding_model