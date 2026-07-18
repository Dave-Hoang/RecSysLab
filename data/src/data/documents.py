from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from src.config import PROCESSED_MOVIES_PATH


REQUIRED_DOCUMENT_COLUMNS = {
    "movieId",
    "title",
    "genres",
    "rating_mean",
    "rating_count",
    "metadata_text",
}


def load_processed_movies(
    parquet_path: Path = PROCESSED_MOVIES_PATH,
) -> pd.DataFrame:
    """
    Đọc dữ liệu phim đã xử lý từ file Parquet.

    Args:
        parquet_path: Đường dẫn file movies_processed.parquet.

    Returns:
        DataFrame phim đã xử lý.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        ValueError: Nếu file thiếu các cột cần thiết.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(
            "Không tìm thấy file dữ liệu đã xử lý: "
            f"{parquet_path}\n"
            "Hãy chạy preprocessing trước khi build FAISS index."
        )

    movies = pd.read_parquet(parquet_path)

    missing_columns = REQUIRED_DOCUMENT_COLUMNS.difference(
        movies.columns
    )

    if missing_columns:
        raise ValueError(
            "File Parquet thiếu các cột cần thiết để tạo Document: "
            f"{sorted(missing_columns)}"
        )

    return movies


def create_documents(
    movies: pd.DataFrame,
) -> list[Document]:
    """
    Chuyển DataFrame phim thành danh sách LangChain Document.

    page_content:
        metadata_text gồm title, genres và tags.

    metadata:
        - movieId
        - title
        - genres
        - rating_mean
        - rating_count

    Args:
        movies: DataFrame phim đã xử lý.

    Returns:
        Danh sách LangChain Document.
    """
    missing_columns = REQUIRED_DOCUMENT_COLUMNS.difference(
        movies.columns
    )

    if missing_columns:
        raise ValueError(
            "Movies DataFrame thiếu các cột cần thiết: "
            f"{sorted(missing_columns)}"
        )

    documents: list[Document] = []

    for row in movies.itertuples(index=False):
        metadata_text = getattr(row, "metadata_text", "")

        if pd.isna(metadata_text) or not str(metadata_text).strip():
            metadata_text = f"Title: {getattr(row, 'title', '')}"

        document = Document(
            page_content=str(metadata_text).strip(),
            metadata={
                "movieId": int(row.movieId),
                "title": str(row.title),
                "genres": str(row.genres),
                "rating_mean": float(row.rating_mean),
                "rating_count": int(row.rating_count),
            },
        )

        documents.append(document)

    return documents


def load_documents(
    parquet_path: Path = PROCESSED_MOVIES_PATH,
) -> list[Document]:
    """
    Đọc Parquet rồi chuyển trực tiếp thành Documents.

    Args:
        parquet_path: Đường dẫn file dữ liệu đã xử lý.

    Returns:
        Danh sách LangChain Document.
    """
    movies = load_processed_movies(parquet_path)
    return create_documents(movies)