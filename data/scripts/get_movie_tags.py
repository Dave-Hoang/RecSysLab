#!/usr/bin/env python
"""
Script để lấy tags của một bộ phim bất kỳ từ dữ liệu đã xử lý.
"""

import sys
from pathlib import Path
import pandas as pd

# Đường dẫn file Parquet đã xử lý
PROCESSED_MOVIES_PATH = Path(__file__).resolve().parents[1] / "ml-32m" / "movies_processed.parquet"


def get_movie_tags(movie_identifier: str) -> dict | None:
    """
    Lấy tags của một bộ phim.

    Args:
        movie_identifier: Có thể là movieId (int) hoặc title (str)

    Returns:
        Dictionary chứa thông tin phim và tags, hoặc None nếu không tìm thấy.
    """
    if not PROCESSED_MOVIES_PATH.exists():
        print(f"[LỖI] Không tìm thấy file dữ liệu: {PROCESSED_MOVIES_PATH}")
        print("Hãy chạy preprocessing trước: python -m src.data.preprocessing")
        return None

    df = pd.read_parquet(PROCESSED_MOVIES_PATH)

    # Thử tìm bằng movieId trước
    try:
        movie_id = int(movie_identifier)
        result = df[df["movieId"] == movie_id]
    except ValueError:
        # Nếu không phải số, tìm theo title (khớp một phần, không phân biệt hoa thường)
        result = df[df["title"].str.contains(movie_identifier, case=False, na=False)]

    if result.empty:
        print(f"[KHÔNG TÌM THẤY] Phim: '{movie_identifier}'")
        return None

    if len(result) > 1:
        print(f"[TÌM THẤY {len(result)} PHIM] Hiển thị kết quả đầu tiên:")

    movie = result.iloc[0]

    return {
        "movieId": int(movie["movieId"]),
        "title": movie["title"],
        "genres": movie["genres"],
        "rating_mean": float(movie["rating_mean"]),
        "rating_count": int(movie["rating_count"]),
        "tags_text": movie["tags_text"],
        "tags_list": movie["tags_text"].split() if movie["tags_text"] else [],
    }


def main():
    if len(sys.argv) < 2:
        print("Cách dùng: python get_movie_tags.py <movieId hoặc title>")
        print("Ví dụ:")
        print("  python get_movie_tags.py 1")
        print("  python get_movie_tags.py \"Toy Story\"")
        print("  python get_movie_tags.py \"Inception\"")
        sys.exit(1)

    identifier = " ".join(sys.argv[1:])
    result = get_movie_tags(identifier)

    if result:
        print(f"\n{'='*60}")
        print(f"PHIM: {result['title']} (ID: {result['movieId']})")
        print(f"THỂ LOẠI: {result['genres']}")
        print(f"ĐIỂM TB: {result['rating_mean']:.2f} ({result['rating_count']} lượt đánh giá)")
        print(f"{'='*60}")
        print(f"TAGS ({len(result['tags_list'])} tags):")
        print(f"{'-'*60}")

        if result['tags_list']:
            # In tags theo hàng, 5 tags mỗi hàng
            for i, tag in enumerate(result['tags_list']):
                if i > 0 and i % 5 == 0:
                    print()
                print(f"  #{tag}", end="")
            print()
        else:
            print("  (Không có tags)")

        print(f"\n{'='*60}")
        print(f"Raw tags_text: {result['tags_text']}")


if __name__ == "__main__":
    main()