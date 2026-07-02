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

if __name__ == "__main__":
    build_and_persistence_pipeline()