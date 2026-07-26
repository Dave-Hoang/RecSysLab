from pathlib import Path
import pandas as pd
import re
from langchain_core.documents import Document
import torch
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "ml-32m"

MOVIES_PATH = DATA_DIR / "movies.csv"
TAGS_PATH = DATA_DIR / "tags.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"
OUTPUT_PATH = DATA_DIR / "movies_processed.parquet"


def load_data():
    movies = pd.read_csv(MOVIES_PATH)
    tags = pd.read_csv(TAGS_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    return movies, tags, ratings


def clean_tag(tag):
    tag = str(tag).lower().strip()

    if tag.startswith("http") or "www." in tag:
        return ""

    tag = re.sub(r"[^a-z0-9\s\-\']", " ", tag)

    tag = re.sub(r"\s+", " ", tag).strip()

    if len(tag) < 2:
        return ""

    if len(tag.split()) > 5:
        return ""

    return tag


def aggregate_tags(tags, top_k=40, min_count=2):
    tags = tags.copy()

    tags["tag"] = tags["tag"].fillna("").apply(clean_tag)
    tags = tags[tags["tag"] != ""]

    tag_counts = (
        tags.groupby(["movieId", "tag"])
        .size()
        .reset_index(name="tag_count")
    )

    # nếu một phim có tag chỉ xuất hiện 1 lần thì có thể khá nhiễu
    tag_counts = tag_counts[tag_counts["tag_count"] >= min_count]

    # lấy top_k tag nhiều nhất cho mỗi phim
    tag_counts = tag_counts.sort_values(
        ["movieId", "tag_count", "tag"],
        ascending=[True, False, True]
    )

    top_tags = (
        tag_counts.groupby("movieId")
        .head(top_k)
        .groupby("movieId")["tag"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )

    top_tags = top_tags.rename(columns={"tag": "tags_text"})

    return top_tags


def aggregate_ratings(ratings):
    ratings_agg = (
        ratings.groupby("movieId")["rating"]
        .agg(rating_mean="mean", rating_count="count")
        .reset_index()
    )

    return ratings_agg


def build_metadata_text(row):
    title = row["title"]
    genres = str(row["genres"]).replace("|", " ")
    tags = str(row.get("tags_text", "")).strip()

    if tags and tags != "nan":
        return f"""Title: {title}
Genres: {genres}
Tags: {tags}
""".strip()

    return f"""Title: {title}
Genres: {genres}
""".strip()


def create_documents(df):
    documents = []

    for _, row in df.iterrows():
        doc = Document(
            page_content=row["metadata_text"],
            metadata={
                "movieId": int(row["movieId"]),
                "title": row["title"],
                "genres": row["genres"],
                "rating_mean": float(row["rating_mean"]),
                "rating_count": int(row["rating_count"]),
            },
        )
        documents.append(doc)

    return documents


def main():
    movies, tags, ratings = load_data()

    print("Movies shape:", movies.shape)
    print("Tags shape:", tags.shape)
    print("Ratings shape:", ratings.shape)

    tags_agg = aggregate_tags(tags, top_k=40, min_count=2)
    ratings_agg = aggregate_ratings(ratings)

    movies_processed = (
        movies
        .merge(tags_agg, on="movieId", how="left")
        .merge(ratings_agg, on="movieId", how="left")
    )

    movies_processed["tags_text"] = movies_processed["tags_text"].fillna("")
    movies_processed["rating_mean"] = movies_processed["rating_mean"].fillna(0)
    movies_processed["rating_count"] = movies_processed["rating_count"].fillna(0)

    movies_processed["metadata_text"] = movies_processed.apply(
        build_metadata_text,
        axis=1
    )

    movies_filtered = movies_processed[movies_processed["rating_count"] > 50].copy()

    movies_filtered["metadata_length"] = movies_filtered["metadata_text"].str.len()

    print("Processed movies:", movies_filtered.shape)
    print("Metadata length summary:")
    print(movies_filtered["metadata_length"].describe())

    print(
        movies_filtered[["title", "metadata_length", "metadata_text"]]
        .sort_values("metadata_length", ascending=False)
        .head(5)
    )

    movies_filtered.to_parquet(OUTPUT_PATH, index=False)

    documents = create_documents(movies_filtered)

    print("Number of LangChain documents:", len(documents))
    print("Sample document:")
    print(documents[0])


def load_bge_m3_model():
    print("LOAD MODEL BGE-M3")
    
    # 1. Tự động kiểm tra phần cứng để tối ưu tốc độ chạy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Hệ thống sẽ sử dụng phần cứng: {device.upper()}")
    
    # 2. Định nghĩa tên mô hình trên Hugging Face
    model_name = "BAAI/bge-m3"
    
    # 3. Cấu hình các tham số chạy mô hình
    model_kwargs = {'device': device}
    
    # BGE-M3 khuyên dùng phép toán normalize_embeddings=True 
    # để đưa các vector về dạng chuẩn (độ dài = 1), giúp tính toán Cosine Similarity chính xác hơn
    encode_kwargs = {'normalize_embeddings': True}
    
    print(f"[*] Đang kết nối và nạp model '{model_name}'...")
    print("[LƯU Ý]: Nếu là lần đầu chạy, tiến trình tải khoảng 2.2GB dữ liệu từ Hugging Face sẽ diễn ra ngầm. Vui lòng đợi trong vài phút...")
    
    # 4. Gọi LangChain khởi tạo mô hình
    bge_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    print("=== KHỞI TẠO MODEL BGE-M3 THÀNH CÔNG! ===")
    return bge_embeddings

if __name__ == "__main__":
    # Chạy thử nghiệm hàm load model
    embedding_engine = load_bge_m3_model()
    
    # Test thử xem model đã hoạt động thực tế chưa bằng cách embed một câu đơn giản
    test_text = "Hệ thống gợi ý phim dựa trên đồ thị và RAG."
    print(f"\n[*] Đang chạy thử nghiệm tạo vector cho câu: '{test_text}'...")
    
    vector_output = embedding_engine.embed_query(test_text)
    
    print(f"[✓] Đã tạo xong Vector thành công!")
    print(f" -> Số chiều của Vector (Dimension): {len(vector_output)}")
    print(f" -> Xem thử 5 phần số đầu tiên trong vector: {vector_output[:5]}")


###############################################################################


import os
import torch
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

# 1. Định nghĩa các đường dẫn file 
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MOVIES_PARQUET_PATH = PROJECT_DIR / "data" / "ml-32m" / "movies_processed.parquet"
FAISS_SAVE_DIR = PROJECT_DIR / "data" / "faiss_movie_index"

def create_documents_from_parquet(parquet_path):
    print(f"[*] Đang đọc dữ liệu phim từ file Parquet: {parquet_path}")
    df = pd.read_parquet(MOVIES_PARQUET_PATH)
    
    # Để kiểm tra nhanh luồng chạy (PoC), ta có thể lấy thử 1000 dòng đầu tiên
    # Sau khi chạy mượt mà, bạn có thể xóa bớt đoạn [.head(1000)] để chạy toàn bộ bảng
    df_subset = df
    
    documents = []
    for _, row in df_subset.iterrows():
        # Kiểm tra nếu cột text bị trống thì bỏ qua hoặc điền chuỗi rỗng
        text_content = row.get("metadata_text", "")
        if not text_content or pd.isna(text_content):
            text_content = str(row.get("title", "")) # Fallback về tên phim nếu trống
            
        doc = Document(
            page_content=str(text_content),
            metadata={
                "movieId": int(row["movieId"]),
                "title": str(row["title"]),
                "genres": str(row["genres"]),
                "rating_mean": float(row.get("rating_mean", 0.0)),
                "rating_count": int(row.get("rating_count", 0)),
            },
        )
        documents.append(doc)
        
    print(f"[✓] Đã chuyển đổi xong {len(documents)} bộ phim thành dạng Documents.")
    return documents

def build_and_persistence_pipeline():
    # Bước 1: Khởi tạo mô hình embedding chuẩn hóa (BGE-M3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True} # Rất quan trọng để tối ưu toán học
    )
    
    # Bước 2: Tạo danh sách tài liệu từ file lưu trữ Parquet của ngày 1
    documents = create_documents_from_parquet(MOVIES_PARQUET_PATH)
    
    # Bước 3 & 4: Tiến hành Embedding hàng loạt và dựng cấu trúc FAISS Store trên RAM
    print("\n[*] Đang tiến hành tạo Embedding và nạp vào cấu trúc FAISS...")
    print("[LƯU Ý]: Quá trình này chạy trên GPU CUDA nên sẽ rất nhanh, hãy đợi trong giây lát...")
    
    # Sử dụng phép toán đo khoảng cách vô hướng (Inner Product) 
    # Kết hợp với normalize_embeddings=True phía trên sẽ tương đương với Cosine Similarity
    vector_store = FAISS.from_documents(documents, embedding_model)
    
    # Bước 5: Lưu trữ thư mục Index xuống local để tái sử dụng
    print(f"\n[*] Đang xuất và đóng gói dữ liệu xuống thư mục: {FAISS_SAVE_DIR}")
    vector_store.save_local(FAISS_SAVE_DIR)
    print("=== PIPELINE HOÀN THÀNH XUẤT SẮC! KHO LƯU TRỮ ĐÃ SẴN SÀNG ===")


#####################################
import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Định nghĩa đường dẫn tới kho lưu trữ FAISS đã đóng gói
FAISS_INDEX_PATH = "D:/Project/RecSysLab/data/faiss_movie_index"

def init_search_engine():
    print("[*] Đang khởi tạo thước đo Embedding (BGE-M3)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Cấu hình phải đồng nhất hoàn toàn với lúc build index (bắt buộc phải normalize)
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("[*] Đang nạp cấu trúc FAISS Index từ ổ đĩa lên RAM...")
    # Cho phép allow_dangerous_deserialization=True để LangChain giải nén file index.pkl
    db = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    print("[✓] HỆ THỐNG TRUY VẤN ĐÃ SẴN SÀNG!\n")
    return db

def k_nearest_neighbors_search(db, query_text, k=10):
    print(f"\n Đang tìm kiếm Top {k} phim phù hợp cho yêu cầu: '{query_text}'...")
    
    # Thực hiện tìm kiếm tương đồng ngữ nghĩa
    results = db.similarity_search(query_text, k=k)
    
    # Duyệt qua kết quả và hiển thị trực quan
    for rank, doc in enumerate(results, 1):
        print(f"\n[Top {rank}] - {doc.metadata['title']}")
        print(f" └─ Thể loại      : {doc.metadata['genres']}")
        print(f" └─ Điểm số (Mean): {doc.metadata['rating_mean']:.2f} ({doc.metadata['rating_count']} lượt đánh giá)")
        print(f" └─ Text ngữ cảnh : {doc.page_content}")

if __name__ == "__main__":
    # 1. Khởi động engine (Chỉ mất vài giây để load lại index cũ)
    movie_search_db = init_search_engine()
    
    # 2. Tiến hành test thử nghiệm các kịch bản truy vấn ngữ nghĩa phức tạp
    
    # Kịch bản 1: Tìm kiếm theo đặc trưng nội dung cốt truyện (Tâm lý, du hành thời gian)
    k_nearest_neighbors_search(movie_search_db, "movies like interstellar with space and philosophy", k=10)
    
    # Kịch bản 2: Tìm kiếm theo cảm xúc và đối tượng người xem (Hoạt hình hài hước gia đình)
    k_nearest_neighbors_search(movie_search_db, "romantic comedy about fake relationship", k=10)
    
    # Kịch bản 3: Tìm kiếm theo tông màu phim tăm tối, giật gân
    k_nearest_neighbors_search(movie_search_db, "sad movie about father and son", k=10)




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

if __name__ == "__main__":
    main()