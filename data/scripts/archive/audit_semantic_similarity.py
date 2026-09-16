import os
import sys
import pandas as pd
from pathlib import Path

# Đặt TEST_MODE=true trước khi import bất kỳ module nào của project
os.environ["TEST_MODE"] = "true"

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.vector_store import load_vector_store
from src.retrieval.embeddings import load_embedding_model
from src.ranking.hybrid_ranker import retrieve_movies_with_score, compute_semantic_similarity
from src.graph.nodes import _get_expansion_llm
from src.graph.prompts import EXPANSION_SYSTEM_PROMPT, EXPANSION_USER_TEMPLATE

def test_leakage_and_audit():
    print("Loading models...")
    embedding_model = load_embedding_model()
    vector_store = load_vector_store(embedding_model)
    
    # Check FAISS index type
    print(f"\nFAISS index type: {type(vector_store.index)}")
    
    # Step 1: Check leakage for "Searching" and "Magnolia"
    # Actually, we don't know the exact query that returned "Searching" or "Magnolia" with exactly 1.000 in this script unless we run it, but we can search for those titles in the vector store and inspect their page_content.
    docs = list(vector_store.docstore._dict.values())
    print(f"Total docs in vector store: {len(docs)}")
    
    for doc in docs:
        if "Searching" in doc.metadata.get("title", "") or "Magnolia" in doc.metadata.get("title", ""):
            print(f"\n--- Document Info ---")
            print(f"Title: {doc.metadata.get('title')}")
            print(f"Content:\n{doc.page_content}")

    # Step 2: Audit semantic similarity across 10 queries
    queries = [
        "I want a raw, cathartic film...",
        "I want an emotional movie that will stay with me for days",
        "phim hài nhẹ nhàng cuối tuần",
        "phim hành động giật gân",
        "phim về tình bạn tuổi thơ",
        "phim kinh dị tâm lý",
        "phim khoa học viễn tưởng có yếu tố triết học",
        "A visually stunning sci-fi adventure",
        "A gripping legal thriller with plot twists",
        "A heartwarming romance set in a small town"
    ]
    
    print("\n--- Auditing Semantic Similarity > 0.98 ---")
    high_sim_cases = []
    
    for q in queries:
        candidates = retrieve_movies_with_score(vector_store, q, k=50) # top 50
        df = pd.DataFrame(candidates)
        df = compute_semantic_similarity(df)
        
        for _, row in df.iterrows():
            if row["semantic_similarity"] > 0.98:
                high_sim_cases.append({
                    "query": q,
                    "title": row["title"],
                    "semantic_similarity": row["semantic_similarity"],
                    "faiss_distance": row["faiss_distance"]
                })
                
    if high_sim_cases:
        print(f"Found {len(high_sim_cases)} cases with semantic_similarity > 0.98:")
        df_high_sim = pd.DataFrame(high_sim_cases)
        print(df_high_sim.to_string())
    else:
        print("No cases found with semantic_similarity > 0.98.")

    # Step 4: Check boilerplate in query expansion
    print("\n--- Auditing Query Expansion Boilerplate ---")
    llm = _get_expansion_llm()
    test_queries = [
        "phim hài nhẹ nhàng cuối tuần",
        "phim hành động giật gân",
        "phim về tình bạn tuổi thơ",
        "phim kinh dị tâm lý",
        "phim khoa học viễn tưởng có yếu tố triết học"
    ]
    
    for q in test_queries:
        messages = [
            ("system", EXPANSION_SYSTEM_PROMPT),
            ("human", EXPANSION_USER_TEMPLATE.format(query=q)),
        ]
        response = llm.invoke(messages)
        print(f"\nOriginal: {q}")
        print(f"Expanded: {response.content}")

if __name__ == "__main__":
    test_leakage_and_audit()
