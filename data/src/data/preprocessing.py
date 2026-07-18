import re
from pathlib import Path
from typing import Final

import pandas as pd

from src.config import (
    MIN_RATING_COUNT,
    MOVIES_PATH,
    PROCESSED_MOVIES_PATH,
    RATINGS_PATH,
    TAG_MIN_COUNT,
    TAG_TOP_K,
    TAGS_PATH,
    validate_raw_data_paths,
)


TAG_CLEAN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[^a-z0-9\s\-']"
)

MULTIPLE_SPACES_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")


def load_raw_data(
    movies_path: Path = MOVIES_PATH,
    tags_path: Path = TAGS_PATH,
    ratings_path: Path = RATINGS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        Tuple gồm:
        - movies DataFrame
        - tags DataFrame
        - ratings DataFrame
    """
    if (
        movies_path == MOVIES_PATH
        and tags_path == TAGS_PATH
        and ratings_path == RATINGS_PATH
    ):
        validate_raw_data_paths()

    movies = pd.read_csv(
        movies_path,
        usecols=["movieId", "title", "genres"],
        dtype={
            "movieId": "int32",
            "title": "string",
            "genres": "string",
        },
    )

    tags = pd.read_csv(
        tags_path,
        usecols=["movieId", "tag"],
        dtype={
            "movieId": "int32",
            "tag": "string",
        },
    )

    ratings = pd.read_csv(
        ratings_path,
        usecols=["movieId", "rating"],
        dtype={
            "movieId": "int32",
            "rating": "float32",
        },
    )

    return movies, tags, ratings


def clean_tag(tag: object) -> str:
    """
    Chuẩn hóa một tag MovieLens.

    Quy tắc giữ nguyên từ code cũ:
    - chuyển thành chữ thường;
    - bỏ khoảng trắng đầu cuối;
    - loại URL;
    - chỉ giữ chữ cái tiếng Anh, số, dấu cách, dấu gạch ngang
      và dấu nháy đơn;
    - loại tag dưới 2 ký tự;
    - loại tag dài hơn 5 từ.

    Args:
        tag: Giá trị tag ban đầu.

    Returns:
        Tag đã làm sạch hoặc chuỗi rỗng nếu không hợp lệ.
    """
    if tag is None or pd.isna(tag):
        return ""

    cleaned_tag = str(tag).lower().strip()

    if not cleaned_tag:
        return ""

    if cleaned_tag.startswith("http") or "www." in cleaned_tag:
        return ""

    cleaned_tag = TAG_CLEAN_PATTERN.sub(" ", cleaned_tag)
    cleaned_tag = MULTIPLE_SPACES_PATTERN.sub(" ", cleaned_tag).strip()

    if len(cleaned_tag) < 2:
        return ""

    if len(cleaned_tag.split()) > 5:
        return ""

    return cleaned_tag


def aggregate_tags(
    tags: pd.DataFrame,
    top_k: int = TAG_TOP_K,
    min_count: int = TAG_MIN_COUNT,
) -> pd.DataFrame:
    """
    Tổng hợp các tag phổ biến nhất cho từng phim.

    Pipeline:
        raw tags
        → clean tag
        → đếm số lần xuất hiện của từng tag trên mỗi movieId
        → bỏ tag xuất hiện dưới min_count lần
        → lấy tối đa top_k tag cho mỗi phim
        → nối thành tags_text

    Args:
        tags: DataFrame gồm movieId và tag.
        top_k: Số tag tối đa giữ lại trên mỗi phim.
        min_count: Số lần xuất hiện tối thiểu của tag trên một phim.

    Returns:
        DataFrame gồm:
        - movieId
        - tags_text
    """
    required_columns = {"movieId", "tag"}
    missing_columns = required_columns.difference(tags.columns)

    if missing_columns:
        raise ValueError(
            "Tags DataFrame thiếu các cột: "
            f"{sorted(missing_columns)}"
        )

    if top_k <= 0:
        raise ValueError("top_k phải lớn hơn 0.")

    if min_count <= 0:
        raise ValueError("min_count phải lớn hơn 0.")

    cleaned_tags = tags.loc[:, ["movieId", "tag"]].copy()

    cleaned_tags["tag"] = cleaned_tags["tag"].apply(clean_tag)

    cleaned_tags = cleaned_tags.loc[
        cleaned_tags["tag"] != ""
    ]

    tag_counts = (
        cleaned_tags
        .groupby(["movieId", "tag"], as_index=False)
        .size()
        .rename(columns={"size": "tag_count"})
    )

    tag_counts = tag_counts.loc[
        tag_counts["tag_count"] >= min_count
    ]

    tag_counts = tag_counts.sort_values(
        by=["movieId", "tag_count", "tag"],
        ascending=[True, False, True],
    )

    top_tags = (
        tag_counts
        .groupby("movieId", group_keys=False)
        .head(top_k)
        .groupby("movieId", as_index=False)["tag"]
        .agg(" ".join)
        .rename(columns={"tag": "tags_text"})
    )

    return top_tags


def aggregate_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Tính rating trung bình và số lượt rating của từng phim.

    Ratings không được đưa vào embedding text. Hai giá trị này
    được sử dụng cho hybrid ranking.

    Args:
        ratings: DataFrame gồm movieId và rating.

    Returns:
        DataFrame gồm:
        - movieId
        - rating_mean
        - rating_count
    """
    required_columns = {"movieId", "rating"}
    missing_columns = required_columns.difference(ratings.columns)

    if missing_columns:
        raise ValueError(
            "Ratings DataFrame thiếu các cột: "
            f"{sorted(missing_columns)}"
        )

    ratings_agg = (
        ratings
        .groupby("movieId", as_index=False)["rating"]
        .agg(
            rating_mean="mean",
            rating_count="count",
        )
    )

    ratings_agg["rating_mean"] = (
        ratings_agg["rating_mean"].astype("float32")
    )

    ratings_agg["rating_count"] = (
        ratings_agg["rating_count"].astype("int32")
    )

    return ratings_agg


def build_metadata_text(row: pd.Series) -> str:
    """
    Xây dựng text dùng để sinh embedding cho một bộ phim.

    Rating không được đưa vào text này.

    Format:
        Title: ...
        Genres: ...
        Tags: ...

    Args:
        row: Một hàng của DataFrame phim đã merge.

    Returns:
        Chuỗi metadata semantic.
    """
    title = str(row.get("title", "")).strip()

    genres_value = row.get("genres", "")
    genres = "" if pd.isna(genres_value) else str(genres_value)
    genres = genres.replace("|", " ").strip()

    tags_value = row.get("tags_text", "")
    tags = "" if pd.isna(tags_value) else str(tags_value).strip()

    metadata_parts = [
        f"Title: {title}",
        f"Genres: {genres}",
    ]

    if tags:
        metadata_parts.append(f"Tags: {tags}")

    return "\n".join(metadata_parts)


def build_processed_movies(
    movies: pd.DataFrame,
    tags: pd.DataFrame,
    ratings: pd.DataFrame,
    tag_top_k: int = TAG_TOP_K,
    tag_min_count: int = TAG_MIN_COUNT,
    min_rating_count: int = MIN_RATING_COUNT,
) -> pd.DataFrame:
    """
    Tạo DataFrame phim đã xử lý hoàn chỉnh.

    Args:
        movies: Dữ liệu movies.csv.
        tags: Dữ liệu tags.csv.
        ratings: Dữ liệu ratings.csv.
        tag_top_k: Số tag tối đa mỗi phim.
        tag_min_count: Số lần xuất hiện tối thiểu của tag.
        min_rating_count: Chỉ giữ phim có rating_count lớn hơn
            giá trị này.

    Returns:
        DataFrame đã merge, tạo metadata và lọc phim.
    """
    required_movie_columns = {"movieId", "title", "genres"}
    missing_movie_columns = required_movie_columns.difference(
        movies.columns
    )

    if missing_movie_columns:
        raise ValueError(
            "Movies DataFrame thiếu các cột: "
            f"{sorted(missing_movie_columns)}"
        )

    tags_agg = aggregate_tags(
        tags=tags,
        top_k=tag_top_k,
        min_count=tag_min_count,
    )

    ratings_agg = aggregate_ratings(ratings)

    processed_movies = (
        movies
        .merge(tags_agg, on="movieId", how="left")
        .merge(ratings_agg, on="movieId", how="left")
    )

    processed_movies["tags_text"] = (
        processed_movies["tags_text"]
        .fillna("")
        .astype("string")
    )

    processed_movies["rating_mean"] = (
        processed_movies["rating_mean"]
        .fillna(0.0)
        .astype("float32")
    )

    processed_movies["rating_count"] = (
        processed_movies["rating_count"]
        .fillna(0)
        .astype("int32")
    )

    processed_movies["metadata_text"] = processed_movies.apply(
        build_metadata_text,
        axis=1,
    )

    filtered_movies = processed_movies.loc[
        processed_movies["rating_count"] > min_rating_count
    ].copy()

    filtered_movies["metadata_length"] = (
        filtered_movies["metadata_text"]
        .str.len()
        .astype("int32")
    )

    filtered_movies = filtered_movies.reset_index(drop=True)

    return filtered_movies


def save_processed_movies(
    movies: pd.DataFrame,
    output_path: Path = PROCESSED_MOVIES_PATH,
) -> None:
    """
    Lưu DataFrame phim đã xử lý thành Parquet.

    Args:
        movies: DataFrame cần lưu.
        output_path: Đường dẫn file Parquet.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    movies.to_parquet(output_path, index=False)


def run_preprocessing(
    output_path: Path = PROCESSED_MOVIES_PATH,
) -> pd.DataFrame:
    """
    Chạy toàn bộ pipeline preprocessing ngày 1.

    Hàm này có thể được gọi từ script riêng, không tự động chạy
    khi module được import.

    Returns:
        DataFrame phim đã xử lý.
    """
    print("[1/4] Đang đọc dữ liệu MovieLens...")
    movies, tags, ratings = load_raw_data()

    print(f"Movies shape : {movies.shape}")
    print(f"Tags shape   : {tags.shape}")
    print(f"Ratings shape: {ratings.shape}")

    print("[2/4] Đang tổng hợp tags và ratings...")
    processed_movies = build_processed_movies(
        movies=movies,
        tags=tags,
        ratings=ratings,
    )

    print(
        f"[3/4] Số phim sau khi lọc: "
        f"{len(processed_movies):,}"
    )

    print("[4/4] Đang lưu file Parquet...")
    save_processed_movies(
        movies=processed_movies,
        output_path=output_path,
    )

    print(f"[✓] Đã lưu dữ liệu tại: {output_path}")

    return processed_movies