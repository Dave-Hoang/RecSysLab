import torch
from langchain_huggingface import HuggingFaceEmbeddings

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