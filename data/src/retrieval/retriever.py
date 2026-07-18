from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import RETRIEVAL_TOP_K


def build_retriever(
    vector_store: FAISS,
    k: int = RETRIEVAL_TOP_K,
    search_type: str = "similarity",
) -> VectorStoreRetriever:
    """
    Chuyển FAISS vector store thành LangChain Retriever.

    Args:
        vector_store:
            FAISS vector store đã load.
        k:
            Số lượng candidates cần lấy.
        search_type:
            Kiểu tìm kiếm, mặc định là similarity.

    Returns:
        VectorStoreRetriever.
    """
    if k <= 0:
        raise ValueError("k phải lớn hơn 0.")

    supported_search_types = {
        "similarity",
        "mmr",
        "similarity_score_threshold",
    }

    if search_type not in supported_search_types:
        raise ValueError(
            f"search_type không hợp lệ: {search_type}. "
            f"Các giá trị hỗ trợ: {sorted(supported_search_types)}"
        )

    search_kwargs: dict[str, Any] = {"k": k}

    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def document_to_candidate(
    document: Document,
    rank: int,
) -> dict:
    """
    Chuyển LangChain Document thành dictionary candidate.

    Args:
        document:
            LangChain Document.
        rank:
            Vị trí ban đầu trong retrieval results.

    Returns:
        Dictionary chứa thông tin phim.
    """
    metadata = document.metadata

    return {
        "rank": int(rank),
        "movieId": int(metadata.get("movieId", -1)),
        "title": str(metadata.get("title", "")),
        "genres": str(metadata.get("genres", "")),
        "rating_mean": float(
            metadata.get("rating_mean", 0.0)
        ),
        "rating_count": int(
            metadata.get("rating_count", 0)
        ),
        "page_content": document.page_content,
    }


def documents_to_candidates(
    documents: list[Document],
) -> list[dict]:
    """
    Chuyển danh sách Documents thành danh sách candidate dictionaries.
    """
    return [
        document_to_candidate(document, rank=index)
        for index, document in enumerate(documents, start=1)
    ]


def retrieve_movies(
    retriever: VectorStoreRetriever,
    query: str,
) -> list[dict]:
    """
    Lấy movie candidates thông qua LangChain Retriever.

    Hàm này không trả về FAISS distance.

    Args:
        retriever:
            Retriever đã được build.
        query:
            Truy vấn ngôn ngữ tự nhiên của người dùng.

    Returns:
        Danh sách candidates.
    """
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Query không được để trống.")

    documents: list[Document] = retriever.invoke(cleaned_query)

    return documents_to_candidates(documents)


def retrieve_movies_with_score(
    vector_store: FAISS,
    query: str,
    k: int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """
    Lấy movie candidates cùng FAISS distance.

    Lưu ý:
        Giá trị trả về từ similarity_search_with_score() của FAISS
        trong cấu hình hiện tại nên được hiểu là distance:

            distance thấp hơn → kết quả gần query hơn.

        Không gọi giá trị này là semantic_score vì tên đó dễ gây
        hiểu nhầm rằng score càng cao càng tốt.

    Args:
        vector_store:
            FAISS vector store.
        query:
            Truy vấn của người dùng.
        k:
            Số candidates.

    Returns:
        Danh sách candidate dictionaries có thêm:
        - faiss_distance
    """
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Query không được để trống.")

    if k <= 0:
        raise ValueError("k phải lớn hơn 0.")

    results = vector_store.similarity_search_with_score(
        query=cleaned_query,
        k=k,
    )

    candidates: list[dict] = []

    for rank, (document, distance) in enumerate(
        results,
        start=1,
    ):
        candidate = document_to_candidate(
            document=document,
            rank=rank,
        )

        candidate["faiss_distance"] = float(distance)

        candidates.append(candidate)

    return candidates