from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from src.config import CROSS_ENCODER_MODEL_NAME


_cross_encoder: CrossEncoder | None = None


def get_cross_encoder_device() -> str:
    """
    Xác định thiết bị chạy Cross-Encoder.

    Returns:
        'cuda' nếu CUDA khả dụng, ngược lại là 'cpu'.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_cross_encoder(
    model_name: str = CROSS_ENCODER_MODEL_NAME,
    device: str | None = None,
) -> CrossEncoder:
    """
    Khởi tạo Cross-Encoder.

    Model chỉ nên được load một lần khi ứng dụng bắt đầu,
    không được load lại ở mỗi query.

    Args:
        model_name:
            Tên Cross-Encoder model.
        device:
            'cuda', 'cpu', hoặc None để tự động phát hiện.

    Returns:
        CrossEncoder đã được khởi tạo.
    """
    selected_device = device or get_cross_encoder_device()

    print(f"[*] Cross-Encoder model: {model_name}")
    print(f"[*] Device: {selected_device.upper()}")

    model = CrossEncoder(
        model_name_or_path=model_name,
        device=selected_device,
        activation_fn=torch.nn.Sigmoid(),
    )

    print("[✓] Load Cross-Encoder thành công.")

    return model


def get_cross_encoder(
    model_name: str = CROSS_ENCODER_MODEL_NAME,
) -> CrossEncoder:
    """
    Trả về singleton Cross-Encoder dùng chung trong process.

    Lần gọi đầu tiên sẽ load model.
    Các lần sau tái sử dụng model đã có.
    """
    global _cross_encoder

    if _cross_encoder is None:
        _cross_encoder = load_cross_encoder(
            model_name=model_name,
        )

    return _cross_encoder


def predict_relevance_scores(
    query: str,
    documents: Sequence[str],
    cross_encoder: CrossEncoder | None = None,
) -> np.ndarray:
    """
    Chấm điểm relevance cho các cặp query-document.

    Cross-Encoder chỉ được dùng trên Top-K candidates từ FAISS,
    không quét toàn bộ corpus.

    Args:
        query:
            Query của người dùng.
        documents:
            Danh sách page_content của các candidates.
        cross_encoder:
            Model được truyền vào. Nếu None, dùng singleton mặc định.

    Returns:
        NumPy array chứa relevance score.

    Raises:
        ValueError:
            Nếu query trống hoặc không có document.
    """
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Query không được để trống.")

    document_list = [
        str(document).strip()
        for document in documents
    ]

    if not document_list:
        return np.array([], dtype=np.float32)

    model = cross_encoder or get_cross_encoder()

    pairs = [
        [cleaned_query, document]
        for document in document_list
    ]

    scores = model.predict(
        pairs,
        show_progress_bar=False,
    )

    return np.asarray(scores, dtype=np.float32)