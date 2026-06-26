from pathlib import Path
import pandas as pd
import re
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "ml-32m"

MOVIES_PATH = DATA_DIR / "movies.csv"
TAGS_PATH = DATA_DIR / "tags.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"
OUTPUT_PATH = DATA_DIR / "movies_processed.parquet"


def load_data():
    movies = pd.read_csv(MOVIES_PATH)
    tags = pd.read_csv(TAGS_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    return movies, tags, ratings


def clean_tag(tag):
    tag = str(tag).lower().strip()

    if tag.startswith("http") or "www." in tag:
        return ""

    tag = re.sub(r"[^a-z0-9\s\-\']", " ", tag)

    tag = re.sub(r"\s+", " ", tag).strip()

    if len(tag) < 2:
        return ""

    if len(tag.split()) > 5:
        return ""

    return tag


def aggregate_tags(tags, top_k=40, min_count=2):
    tags = tags.copy()

    tags["tag"] = tags["tag"].fillna("").apply(clean_tag)
    tags = tags[tags["tag"] != ""]

    tag_counts = (
        tags.groupby(["movieId", "tag"])
        .size()
        .reset_index(name="tag_count")
    )

    # nếu một phim có tag chỉ xuất hiện 1 lần thì có thể khá nhiễu
    tag_counts = tag_counts[tag_counts["tag_count"] >= min_count]

    # lấy top_k tag nhiều nhất cho mỗi phim
    tag_counts = tag_counts.sort_values(
        ["movieId", "tag_count", "tag"],
        ascending=[True, False, True]
    )

    top_tags = (
        tag_counts.groupby("movieId")
        .head(top_k)
        .groupby("movieId")["tag"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )

    top_tags = top_tags.rename(columns={"tag": "tags_text"})

    return top_tags


def aggregate_ratings(ratings):
    ratings_agg = (
        ratings.groupby("movieId")["rating"]
        .agg(rating_mean="mean", rating_count="count")
        .reset_index()
    )

    return ratings_agg


def build_metadata_text(row):
    title = row["title"]
    genres = str(row["genres"]).replace("|", " ")
    tags = str(row.get("tags_text", "")).strip()

    if tags and tags != "nan":
        return f"""Title: {title}
Genres: {genres}
Tags: {tags}
""".strip()

    return f"""Title: {title}
Genres: {genres}
""".strip()


def create_documents(df):
    documents = []

    for _, row in df.iterrows():
        doc = Document(
            page_content=row["metadata_text"],
            metadata={
                "movieId": int(row["movieId"]),
                "title": row["title"],
                "genres": row["genres"],
                "rating_mean": float(row["rating_mean"]),
                "rating_count": int(row["rating_count"]),
            },
        )
        documents.append(doc)

    return documents


def main():
    movies, tags, ratings = load_data()

    print("Movies shape:", movies.shape)
    print("Tags shape:", tags.shape)
    print("Ratings shape:", ratings.shape)

    tags_agg = aggregate_tags(tags, top_k=40, min_count=2)
    ratings_agg = aggregate_ratings(ratings)

    movies_processed = (
        movies
        .merge(tags_agg, on="movieId", how="left")
        .merge(ratings_agg, on="movieId", how="left")
    )

    movies_processed["tags_text"] = movies_processed["tags_text"].fillna("")
    movies_processed["rating_mean"] = movies_processed["rating_mean"].fillna(0)
    movies_processed["rating_count"] = movies_processed["rating_count"].fillna(0)

    movies_processed["metadata_text"] = movies_processed.apply(
        build_metadata_text,
        axis=1
    )

    movies_filtered = movies_processed[movies_processed["rating_count"] > 50].copy()

    movies_filtered["metadata_length"] = movies_filtered["metadata_text"].str.len()

    print("Processed movies:", movies_filtered.shape)
    print("Metadata length summary:")
    print(movies_filtered["metadata_length"].describe())

    print(
        movies_filtered[["title", "metadata_length", "metadata_text"]]
        .sort_values("metadata_length", ascending=False)
        .head(5)
    )

    movies_filtered.to_parquet(OUTPUT_PATH, index=False)

    documents = create_documents(movies_filtered)

    print("Number of LangChain documents:", len(documents))
    print("Sample document:")
    print(documents[0])


if __name__ == "__main__":
    main()