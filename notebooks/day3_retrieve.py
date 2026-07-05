import torch
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def load_embedding_model(model_name: str = "BAAI/bge-m3") -> HuggingFaceEmbeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Dùng device: {device.upper()}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

def load_vectorstore(faiss_dir: str, embedding_model) -> FAISS:
    print(f"[*] Đang load FAISS index từ: {faiss_dir}")
    db = FAISS.load_local(
        faiss_dir,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    print("[✓] Load thành công!")
    return db

def build_retriever(vectorstore: FAISS, k: int = 20):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

def retrieve_movies(retriever, query: str) -> list[dict]:
    docs: list[Document] = retriever.invoke(query)
    return _docs_to_dicts(docs)

def retrieve_movies_with_score(vectorstore: FAISS, query: str, k: int = 20) -> list[dict]:
    results = vectorstore.similarity_search_with_score(query, k=k)
    candidates = []
    for rank, (doc, score) in enumerate(results, start=1):
        candidate = _doc_to_dict(doc, rank=rank)
        candidate["semantic_score"] = round(float(score), 6)
        candidates.append(candidate)
    return candidates

def _doc_to_dict(doc: Document, rank: int = None) -> dict:
    d = {}
    if rank is not None:
        d["rank"] = rank
    d.update({
        "movieId": doc.metadata.get("movieId", -1),
        "title": doc.metadata.get("title", ""),
        "genres": doc.metadata.get("genres", ""),
        "rating_mean": doc.metadata.get("rating_mean", 0.0),
        "rating_count": doc.metadata.get("rating_count", 0),
        "page_content": doc.page_content,
    })
    return d

def _docs_to_dicts(docs: list[Document]) -> list[dict]:
    return [_doc_to_dict(doc, rank=i + 1) for i, doc in enumerate(docs)]

if __name__ == "__main__":
    FAISS_DIR = r"D:\Project\RecSysLab\data\faiss_movie_index"
    OUTPUT_CSV = r"D:\Project\RecSysLab\data\retrieval_test_results.csv"

    model = load_embedding_model()
    vs = load_vectorstore(FAISS_DIR, model)
    movie_retriever = build_retriever(vs, k=20)

    test_query = "sad movie about father and son"
    print(f"\n[*] Đang chạy truy vấn: '{test_query}'")

    docs_only = retrieve_movies(movie_retriever, test_query)
    print(f"[✓] Retriever.invoke() trả về {len(docs_only)} documents")

    movie_candidates = retrieve_movies_with_score(vs, test_query, k=20)
    df_results = pd.DataFrame(movie_candidates)

    print("\n[✓] TOP 5 ỨNG VIÊN:")
    print(df_results[["rank", "title", "genres", "semantic_score"]].head(5).to_string(index=False))

    df_results.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[✓] Đã xuất CSV: {OUTPUT_CSV}")