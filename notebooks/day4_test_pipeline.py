# day4_reranking.py — bản đầy đủ để chạy được

import torch
import numpy as np
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# ────────────────────────────────────────
# PHẦN NGÀY 3 — Load model & FAISS
# ────────────────────────────────────────
_cross_encoder = None

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

def retrieve_movies_with_score(vectorstore: FAISS, query: str, k: int = 20) -> list[dict]:
    results = vectorstore.similarity_search_with_score(query, k=k)
    candidates = []
    for rank, (doc, score) in enumerate(results, start=1):
        candidate = {
            "rank": rank,
            "movieId": doc.metadata.get("movieId", -1),
            "title": doc.metadata.get("title", ""),
            "genres": doc.metadata.get("genres", ""),
            "rating_mean": doc.metadata.get("rating_mean", 0.0),
            "rating_count": doc.metadata.get("rating_count", 0),
            "page_content": doc.page_content,
        }
        candidate["semantic_score"] = round(float(score), 6)
        candidates.append(candidate)
    return candidates


# ────────────────────────────────────────
# PHẦN NGÀY 4 — Hybrid Reranking
# ────────────────────────────────────────
def normalize_and_invert_scores(df_candidates: pd.DataFrame) -> pd.DataFrame:
    df = df_candidates.copy()
    if df.empty:
        return df
    df["raw_similarity"] = 1.0 - (df["semantic_score"] / 2.0)
    min_sim = df["raw_similarity"].min()
    max_sim = df["raw_similarity"].max()
    if max_sim - min_sim > 0:
        df["semantic_similarity"] = (df["raw_similarity"] - min_sim) / (max_sim - min_sim)
    else:
        df["semantic_similarity"] = 1.0
    df["semantic_similarity"] = df["semantic_similarity"].round(6)
    return df.drop(columns=["raw_similarity"])

def compute_popularity_score(df_candidates: pd.DataFrame) -> pd.DataFrame:
    df = df_candidates.copy()
    if df.empty:
        return df
    df["raw_popularity"] = df["rating_mean"] * np.log1p(df["rating_count"])
    min_pop = df["raw_popularity"].min()
    max_pop = df["raw_popularity"].max()
    if max_pop - min_pop > 0:
        df["popularity_score"] = (df["raw_popularity"] - min_pop) / (max_pop - min_pop)
    else:
        df["popularity_score"] = 1.0
    df["popularity_score"] = df["popularity_score"].round(6)
    return df.drop(columns=["raw_popularity"])

RULES = {
    "sad": {"boost": ["Drama"], "penalty": ["Comedy", "Horror"]},
    "emotional": {"boost": ["Drama"], "penalty": ["Comedy", "Horror"]},
    "father": {"boost": ["Drama"], "penalty": ["Horror"]},
    "scary": {"boost": ["Horror", "Thriller"], "penalty": []},
    "ghost": {"boost": ["Horror", "Thriller"], "penalty": []},
    "romantic": {"boost": ["Romance", "Comedy"], "penalty": ["Horror"]},
}

def compute_rule_score(df_candidates: pd.DataFrame, query: str) -> pd.DataFrame:
    df = df_candidates.copy()
    if df.empty:
        return df
    query_lower = query.lower()
    STEP = 0.10
    rule_scores = []
    for _, row in df.iterrows():
        genres = str(row.get("genres", ""))
        boost_genres, penalty_genres = set(), set()
        for keyword, rule in RULES.items():
            if keyword in query_lower:
                boost_genres.update(rule["boost"])
                penalty_genres.update(rule["penalty"])
        score = 0.0
        for g in boost_genres:
            if g in genres:
                score += STEP
        for g in penalty_genres:
            if g in genres:
                score -= STEP
        rule_scores.append(score)
    df["rule_score"] = pd.Series(rule_scores).clip(-0.3, 0.3).round(6)
    return df


def get_cross_encoder(model_name: str = "BAAI/bge-reranker-v2-m3") -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print("[*] Đang load Cross-Encoder model...")
        _cross_encoder = CrossEncoder(
            model_name,
            device = "cuda" if torch.cuda.is_available() else "cpu",
            activation_fn=torch.nn.Sigmoid()
        )
        print("[✓] Load Cross-Encoder thành công!")
    return _cross_encoder


# Cross-Encoder scoring
def compute_cross_encoder_score(df_candidates: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Dùng Cross-Encoder để chấm điểm chính xác từng cặp (query, document).
    Chỉ chạy trên Top-K candidates, KHÔNG chạy trên toàn bộ corpus.
    """
    df = df_candidates.copy()
    if df.empty:
        return df

    cross_encoder = get_cross_encoder()
    pairs = [[query, content] for content in df["page_content"]]
    scores = cross_encoder.predict(pairs)

    df["cross_encoder_score"] = scores
    df["cross_encoder_score"] = df["cross_encoder_score"].round(6)

    return df


# Final score
def compute_final_score(
    df_candidates: pd.DataFrame,
    weight_cross_encoder: float = 0.50,
    weight_semantic: float = 0.25,
    weight_popularity: float = 0.15,
    weight_rule: float = 0.10
) -> pd.DataFrame:
    df = df_candidates.copy()
    if df.empty:
        return df

    df["final_score"] = (
        weight_cross_encoder * df["cross_encoder_score"]
        + weight_semantic * df["semantic_similarity"]
        + weight_popularity * df["popularity_score"]
        + weight_rule * df["rule_score"]
    )
    df["final_score"] = df["final_score"].round(6)

    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df["final_rank"] = df.index + 1

    return df


# ────────────────────────────────────────
# PIPELINE HOÀN CHỈNH
# ────────────────────────────────────────
def full_reranking_pipeline(vectorstore, query: str, k: int = 20, top_n: int = 5) -> pd.DataFrame:
    candidates = retrieve_movies_with_score(vectorstore, query, k=k)
    df = pd.DataFrame(candidates)
    df = normalize_and_invert_scores(df)
    df = compute_popularity_score(df)
    df = compute_rule_score(df, query=query)
    df = compute_cross_encoder_score(df, query=query)
    df = compute_final_score(df)
    return df.head(top_n)


# ────────────────────────────────────────
# CHẠY THỰC TẾ
# ────────────────────────────────────────
if __name__ == "__main__":
    FAISS_DIR = r"D:\Project\RecSysLab\data\faiss_movie_index"  # sửa đúng đường dẫn của bạn

    model = load_embedding_model()
    vs = load_vectorstore(FAISS_DIR, model)

    test_query = "scary ghost movie"
    top5 = full_reranking_pipeline(vs, test_query, k=20, top_n=5)

    print(f"\n=== TOP 5 SAU RERANK CHO QUERY: '{test_query}' ===")
    print(top5[["final_rank", "title", "genres", "semantic_similarity", "popularity_score", "rule_score", "cross_encoder_score", "final_score"]].to_string(index=False))