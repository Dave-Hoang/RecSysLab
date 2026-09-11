"""Test Groq Integration."""
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.graph.graph import build_graph
from src.retrieval.embeddings import load_embedding_model
from src.retrieval.vector_store import load_vector_store
from src.ranking.cross_encoder import load_cross_encoder

def main():
    print("Loading embedding model...")
    emb = load_embedding_model()
    print("Loading vector store...")
    vs = load_vector_store(embedding_model=emb)
    print("Loading cross encoder...")
    ce = load_cross_encoder()
    print("Building graph...")
    compiled = build_graph(vector_store=vs, cross_encoder=ce)
    print("[OK] Graph compiled successfully!")

    print("=" * 60)
    print("TEST 1: Fast Path (search_movie)")
    print("=" * 60)
    result = compiled.invoke({
        "original_query": "Phim Inception",
        "top_n": 3,
        "include_explanation": False,
    })
    print(f'Execution path: {result.get("execution_path")}')
    print(f'Intent: {result.get("intent")}')
    print(f'Reasoning: {result.get("intent_reasoning")}')
    print(f'Ranked movies: {len(result.get("ranked_movies", []))}')

    print("=" * 60)
    print("TEST 2: Quality Path (need_expansion)")
    print("=" * 60)
    result = compiled.invoke({
        "original_query": "Tìm phim buồn để khóc một mình",
        "top_n": 3,
        "include_explanation": False,
    })
    print(f'Execution path: {result.get("execution_path")}')
    print(f'Intent: {result.get("intent")}')
    print(f'Reasoning: {result.get("intent_reasoning")}')
    print(f'Expanded query: {result.get("expanded_query")}')
    print(f'Ranked movies: {len(result.get("ranked_movies", []))}')

    print("=" * 60)
    print("TEST 3: Direct Response (not_recommendation)")
    print("=" * 60)
    result = compiled.invoke({
        "original_query": "Bạn là ai vậy",
        "top_n": 3,
        "include_explanation": False,
    })
    print(f'Execution path: {result.get("execution_path")}')
    print(f'Intent: {result.get("intent")}')
    print(f'Direct Response: {result.get("direct_response")}')

if __name__ == "__main__":
    main()
