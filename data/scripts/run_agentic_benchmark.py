"""Phase 6: Run Agentic Benchmark with Multi-LLM (Groq & Gemini)."""
import sys
import time
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import EVALUATION_QUERIES_PATH, EVALUATION_DIR
from src.graph.graph import build_graph
from src.retrieval.embeddings import load_embedding_model
from src.retrieval.vector_store import load_vector_store
from src.ranking.cross_encoder import load_cross_encoder

BENCHMARK_RESULTS_PATH = EVALUATION_DIR / "agentic_benchmark_results.csv"

def main():
    print("=" * 60)
    print("PHASE 6: AGENTIC BENCHMARK")
    print("=" * 60)
    
    if not EVALUATION_QUERIES_PATH.exists():
        print(f"[!] Lỗi: Không tìm thấy {EVALUATION_QUERIES_PATH}")
        sys.exit(1)
        
    queries_df = pd.read_csv(EVALUATION_QUERIES_PATH)
    print(f"[*] Đã tải {len(queries_df)} queries để đánh giá.")
    
    print("\n[*] Khởi tạo Graph (bao gồm FAISS & Cross-Encoder)...")
    emb = load_embedding_model()
    vs = load_vector_store(embedding_model=emb)
    ce = load_cross_encoder()
    compiled_graph = build_graph(vector_store=vs, cross_encoder=ce)
    
    results = []
    
    for idx, row in queries_df.iterrows():
        qid = row.get("query_id", f"Q{idx}")
        category = row.get("category", "")
        difficulty = row.get("difficulty", "")
        original_query = row.get("query", "")
        
        print(f"[{idx+1}/{len(queries_df)}] Đang xử lý: {original_query}")
        
        start_time = time.perf_counter()
        
        inputs = {
            "original_query": original_query,
            "top_n": 5,
            "include_explanation": False, # Tiết kiệm quota Node 7
        }
        
        try:
            output = compiled_graph.invoke(inputs)
            
            elapsed = time.perf_counter() - start_time
            
            intent = output.get("intent", "")
            reasoning = output.get("intent_reasoning", "")
            expanded_query = output.get("expanded_query", "")
            confidence = output.get("confidence_level", "")
            exec_path = " -> ".join(output.get("execution_path", []))
            
            mean_ce = output.get("mean_cross_encoder_score", None)
            mean_sem = output.get("mean_semantic_score", None)
            
            ranked = output.get("ranked_movies")
            top_movies = ""
            if ranked is not None and not ranked.empty:
                top_movies = " | ".join(ranked["title"].astype(str).tolist()[:3])
            
            res = {
                "query_id": qid,
                "category": category,
                "difficulty": difficulty,
                "query": original_query,
                "intent": intent,
                "execution_path": exec_path,
                "latency_s": round(elapsed, 2),
                "confidence_level": confidence,
                "mean_semantic_score": mean_sem,
                "mean_ce_score": mean_ce,
                "top_movies": top_movies,
            }
            results.append(res)
            
        except Exception as e:
            print(f"[!] Lỗi tại query {qid}: {str(e)}")
            results.append({
                "query_id": qid,
                "category": category,
                "difficulty": difficulty,
                "query": original_query,
                "intent": "ERROR",
                "execution_path": "ERROR",
                "latency_s": 0,
                "confidence_level": "",
                "mean_semantic_score": None,
                "mean_ce_score": None,
                "top_movies": str(e),
            })
            
    print("\n[*] Hoàn thành chạy benchmark!")
    res_df = pd.DataFrame(results)
    res_df.to_csv(BENCHMARK_RESULTS_PATH, index=False, encoding="utf-8-sig")
    print(f"[✓] Đã lưu kết quả tại: {BENCHMARK_RESULTS_PATH}")
    
    print("\nTóm tắt Latency:")
    print(res_df["latency_s"].describe())
    
    print("\nThống kê Intent:")
    print(res_df["intent"].value_counts())

if __name__ == "__main__":
    main()
