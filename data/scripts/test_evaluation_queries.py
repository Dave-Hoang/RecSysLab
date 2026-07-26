from collections import Counter

from src.evaluation.query_loader import (
    load_evaluation_queries,
)


def main() -> None:
    queries = load_evaluation_queries()

    print(f"[✓] Đã load {len(queries)} evaluation queries.")

    category_counts = Counter(
        query.category.value
        for query in queries
    )

    difficulty_counts = Counter(
        query.difficulty.value
        for query in queries
    )

    print("\nSố lượng theo category:")

    for category, count in sorted(category_counts.items()):
        print(f"- {category}: {count}")

    print("\nSố lượng theo difficulty:")

    for difficulty, count in sorted(
        difficulty_counts.items()
    ):
        print(f"- {difficulty}: {count}")

    print("\nCác query mẫu:")

    for item in queries[:5]:
        print(
            f"- {item.query_id}: {item.query}\n"
            f"  category={item.category.value}, "
            f"difficulty={item.difficulty.value}\n"
            f"  expected_focus={item.expected_focus}"
        )


if __name__ == "__main__":
    main()