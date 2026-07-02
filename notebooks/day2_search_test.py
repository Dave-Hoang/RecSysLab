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