import os
import sys
from pathlib import Path
import numpy as np

# Đặt TEST_MODE=true trước khi import
os.environ["TEST_MODE"] = "true"
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.vector_store import load_vector_store
from src.retrieval.embeddings import load_embedding_model

def audit_raw_score():
    embedding_model = load_embedding_model()
    vector_store = load_vector_store(embedding_model)
    
    query = "I want a raw, cathartic film..."
    query_vector = embedding_model.embed_query(query)
    
    # Check query norm
    query_norm = np.linalg.norm(query_vector)
    print(f"Query vector L2 norm: {query_norm:.6f}")
    
    # Get top 1 result
    results = vector_store.similarity_search_with_score(query, k=1)
    doc, faiss_dist = results[0]
    
    # Get document vector
    # We need to get the vector from FAISS directly
    doc_id_in_faiss = None
    for k, v in vector_store.docstore._dict.items():
        if v.page_content == doc.page_content:
            doc_id = k
            break
            
    # In langchain community FAISS, index_to_docstore_id maps index to doc_id
    index_id = None
    for i, d_id in vector_store.index_to_docstore_id.items():
        if d_id == doc_id:
            index_id = i
            break
            
    doc_vector = vector_store.index.reconstruct(index_id)
    doc_norm = np.linalg.norm(doc_vector)
    print(f"Doc vector L2 norm: {doc_norm:.6f}")
    
    # Compute dot product (cosine similarity since normalized)
    dot_product = np.dot(query_vector, doc_vector)
    print(f"Raw dot product (cosine similarity): {dot_product:.6f}")
    
    # Compute squared L2 distance manually
    manual_l2_sq = np.sum((np.array(query_vector) - doc_vector) ** 2)
    print(f"Manual squared L2 distance: {manual_l2_sq:.6f}")
    
    print(f"FAISS returned distance: {faiss_dist:.6f}")
    
    # Verify the formula
    formula_cos_sim = 1.0 - (faiss_dist / 2.0)
    print(f"Formula (1 - dist/2): {formula_cos_sim:.6f}")

if __name__ == "__main__":
    audit_raw_score()
