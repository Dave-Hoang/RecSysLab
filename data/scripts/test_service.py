from src.services.recommendation_service import (
    RecommendationService,
)


TEST_QUERIES = [
    "psychological sci-fi movies",
    "sad movie about father and son",
]


def print_result(result) -> None:
    print("\n" + "=" * 100)
    print(f"QUERY: {result.query}")
    print("=" * 100)

    print("\nTOP RECOMMENDATIONS:")

    for movie in result.recommendations:
        print(
            f"{movie['final_rank']}. "
            f"{movie['title']} "
            f"— {movie['genres']} "
            f"— final_score={movie['final_score']}"
        )

    print("\nLLM EXPLANATION:")
    print(result.explanation)

    print("\nTIMINGS:")
    for name, seconds in result.timings.items():
        print(f"- {name}: {seconds}s")


def main() -> None:
    print("[1/2] Khởi tạo RecommendationService...")

    service = RecommendationService()

    print("[2/2] Chạy recommendation pipeline...")

    for query in TEST_QUERIES:
        result = service.recommend(
            query=query,
            include_explanation=True,
        )

        print_result(result)


if __name__ == "__main__":
    main()