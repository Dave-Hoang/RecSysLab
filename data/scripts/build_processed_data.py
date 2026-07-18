from src.data.preprocessing import run_preprocessing


def main() -> None:
    processed_movies = run_preprocessing()

    print("\nMetadata length summary:")
    print(
        processed_movies["metadata_length"]
        .describe()
        .to_string()
    )

    print("\n5 phim có metadata dài nhất:")
    print(
        processed_movies[
            ["title", "metadata_length", "metadata_text"]
        ]
        .sort_values(
            by="metadata_length",
            ascending=False,
        )
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()