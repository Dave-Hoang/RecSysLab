import sys
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import EVALUATION_QUERIES_PATH, EVALUATION_DIR, EVALUATION_LABELS_PATH
from src.graph.graph import build_graph
from src.retrieval.embeddings import load_embedding_model
from src.retrieval.vector_store import load_vector_store
from src.ranking.cross_encoder import load_cross_encoder
from src.evaluation.metrics import compute_all_metrics

def main():
    print("Loading models...")
    emb = load_embedding_model()
    vs = load_vector_store(embedding_model=emb)
    ce = load_cross_encoder()
    compiled_graph = build_graph(vector_store=vs, cross_encoder=ce)
    
    queries_df = pd.read_csv(EVALUATION_QUERIES_PATH)
    
    predictions = []
    
    for idx, row in queries_df.iterrows():
        qid = row.get("query_id")
        query = row.get("query")
        category = row.get("category")
        difficulty = row.get("difficulty")
        
        print(f"Processing: {query}")
        
        inputs = {
            "original_query": query,
            "top_n": 5,
            "include_explanation": False
        }
        
        try:
            output = compiled_graph.invoke(inputs)
            ranked = output.get("ranked_movies")
            
            if ranked is not None and not ranked.empty:
                for rank_idx, movie_row in ranked.head(5).iterrows():
                    predictions.append({
                        "query_id": qid,
                        "query": query,
                        "category": category,
                        "difficulty": difficulty,
                        "configuration": "agentic",
                        "rank": rank_idx + 1,
                        "movieId": movie_row["movieId"],
                        "title": movie_row["title"]
                    })
        except Exception as e:
            print(f"Error on {query}: {e}")
            
    preds_df = pd.DataFrame(predictions)
    preds_df.to_csv(EVALUATION_DIR / "agentic_predictions.csv", index=False)
    
    # Merge with labels to get scores
    labels_df = pd.read_csv(EVALUATION_LABELS_PATH)
    scored_df = pd.merge(
        preds_df,
        labels_df[["query_id", "movieId", "relevance"]],
        on=["query_id", "movieId"],
        how="left"
    )
    
    # Fill NaN relevance with 0
    scored_df["relevance"] = scored_df["relevance"].fillna(0)
    
    print("Computing metrics...")
    artifacts = compute_all_metrics(scored_df, top_k=5, expected_query_count=30, expected_configuration_count=1)
    
    agentic_cat = artifacts.by_category
    
    # Load classic metrics
    classic_cat = pd.read_csv(EVALUATION_DIR / "results" / "category_metrics.csv")
    hybrid_ce_cat = classic_cat[classic_cat["configuration"] == "hybrid_with_ce"]
    
    # Compare
    print("\n" + "="*50)
    print("NDCG@5 COMPARISON: Classic (hybrid_with_ce) vs Agentic")
    print("="*50)
    
    categories = queries_df["category"].unique()
    
    for cat in categories:
        classic_row = hybrid_ce_cat[hybrid_ce_cat["category"] == cat]
        agentic_row = agentic_cat[agentic_cat["category"] == cat]
        
        classic_ndcg = classic_row["ndcg_at_5_mean"].values[0] if not classic_row.empty else 0
        agentic_ndcg = agentic_row["ndcg_at_5_mean"].values[0] if not agentic_row.empty else 0
        
        print(f"Category: {cat:20} | Classic: {classic_ndcg:.4f} | Agentic: {agentic_ndcg:.4f}")

if __name__ == "__main__":
    main()
