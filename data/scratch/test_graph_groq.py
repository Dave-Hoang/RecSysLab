import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.graph.graph import build_graph
from src.retrieval.embeddings import load_embedding_model
from src.retrieval.vector_store import load_vector_store
from src.ranking.cross_encoder import load_cross_encoder

def test_graph():
    print("Loading models...")
    emb = load_embedding_model()
    vs = load_vector_store(embedding_model=emb)
    ce = load_cross_encoder()

    print("Building graph...")
    graph = build_graph(vector_store=vs, cross_encoder=ce)

    query = "Tìm cho tôi phim hành động khoa học viễn tưởng hay"
    print(f"\nQuery: {query}")
    
    inputs = {
        "original_query": query,
        "include_explanation": False
    }
    
    print("Running graph...")
    for output in graph.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}':")
            if "intent" in value:
                print(f"  Intent: {value['intent']}")
                print(f"  Reasoning: {value['intent_reasoning']}")
            if "expanded_query" in value:
                print(f"  Expanded query: {value['expanded_query']}")
            if "ranked_movies" in value:
                movies = value["ranked_movies"]
                print(f"  Ranked movies count: {len(movies)}")
                if not movies.empty:
                    print(f"  Top movie: {movies.iloc[0]['title']}")
            print("---")
            
    print("Done!")

if __name__ == "__main__":
    test_graph()
