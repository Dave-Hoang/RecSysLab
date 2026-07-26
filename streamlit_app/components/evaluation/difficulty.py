import pandas as pd
import streamlit as st

from components.evaluation.utils import (
    format_configuration,
    load_difficulty_metrics,
)

DIFFICULTY_ORDER = [
    "easy",
    "medium",
    "hard",
]


DIFFICULTY_NAMES = {
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
}


SUMMARY_COLUMNS = {
    "configuration": "Method",
    "precision_at_5_mean": "Precision@5",
    "ndcg_at_5_mean": "NDCG@5",
    "mrr_at_5_mean": "MRR@5",
    "mean_relevance_at_5_mean": "Mean Relevance",
}


def get_best_configuration(
    difficulty_df: pd.DataFrame,
    metric: str = "ndcg_at_5_mean",
) -> pd.Series:
    """
    Return the best-performing configuration
    for a given difficulty level.
    """

    return difficulty_df.sort_values(metric, ascending=False).iloc[0]


def prettify_difficulty(
    difficulty: str,
) -> str:
    """
    Convert difficulty id into display name.
    """

    return DIFFICULTY_NAMES.get(
        difficulty,
        difficulty,
    )


def build_summary_table(
    difficulty_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build summary dataframe for display.
    """

    summary = (
        difficulty_df[list(SUMMARY_COLUMNS.keys())]
        .rename(columns=SUMMARY_COLUMNS)
        .copy()
    )

    summary["Method"] = summary["Method"].apply(format_configuration)

    return summary


def prepare_difficulty_summary() -> tuple[pd.DataFrame, dict]:
    """
    Load difficulty evaluation results and
    determine the best configuration for
    each difficulty level.

    Returns
    -------
    df
        Raw difficulty dataframe.

    best_models
        Dictionary mapping each difficulty
        to its best-performing configuration.
    """

    df = load_difficulty_metrics()

    best_models = {}

    for difficulty in DIFFICULTY_ORDER:

        difficulty_df = df[df["difficulty"] == difficulty].copy()

        best_models[difficulty] = get_best_configuration(difficulty_df)

    return df, best_models


def render_difficulty_cards(
    best_models: dict,
):
    """
    Render the best-performing model for each
    difficulty level.
    """

    st.subheader("Best Model by Difficulty")

    columns = st.columns(3)

    for index, difficulty in enumerate(DIFFICULTY_ORDER):

        best = best_models[difficulty]

        with columns[index]:

            with st.container(border=True):

                st.markdown(f"#### {prettify_difficulty(difficulty)}")

                st.markdown(f"🏆 **{format_configuration(best['configuration'])}**")

                st.divider()

                metric_col1, metric_col2, metric_col3 = st.columns(3)

                metric_col1.metric(
                    "P@5",
                    f"{best['precision_at_5_mean']:.3f}",
                )

                metric_col2.metric(
                    "NDCG",
                    f"{best['ndcg_at_5_mean']:.3f}",
                )

                metric_col3.metric(
                    "MRR",
                    f"{best['mrr_at_5_mean']:.3f}",
                )


def render_summary_table(
    df: pd.DataFrame,
):
    """
    Render difficulty comparison table.
    """

    st.subheader("Difficulty Summary")

    rows = []

    for difficulty in DIFFICULTY_ORDER:

        difficulty_df = df[df["difficulty"] == difficulty].copy()

        best = get_best_configuration(difficulty_df)

        rows.append(
            {
                "Difficulty": prettify_difficulty(difficulty),
                "Best Model": format_configuration(best["configuration"]),
                "Precision@5": round(
                    best["precision_at_5_mean"],
                    3,
                ),
                "NDCG@5": round(
                    best["ndcg_at_5_mean"],
                    3,
                ),
                "MRR@5": round(
                    best["mrr_at_5_mean"],
                    3,
                ),
            }
        )

    summary = pd.DataFrame(rows)

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True,
    )


def render_difficulty_insights(
    df: pd.DataFrame,
):
    """
    Generate observations from
    difficulty-level evaluation.
    """

    st.subheader("Observations")

    winners = df.groupby("configuration")["ndcg_at_5_mean"].idxmax()

    winner_count = df.loc[winners].groupby("configuration").size()

    overall_best = winner_count.idxmax()

    overall_count = winner_count.max()

    easy = (
        df[df["difficulty"] == "easy"]
        .sort_values(
            "ndcg_at_5_mean",
            ascending=False,
        )
        .iloc[0]
    )

    hard = (
        df[df["difficulty"] == "hard"]
        .sort_values(
            "ndcg_at_5_mean",
            ascending=False,
        )
        .iloc[0]
    )

    st.markdown(f"""
- **{format_configuration(overall_best)}** achieved the best NDCG@5 in **{overall_count}** difficulty levels.

- The strongest performance on **Easy** queries was achieved by **{format_configuration(easy["configuration"])}**.

- The strongest performance on **Hard** queries was achieved by **{format_configuration(hard["configuration"])}**.

- Recommendation quality generally decreases as query difficulty increases, highlighting the importance of semantic retrieval and reranking for challenging recommendation scenarios.
""")


def render_difficulty_metrics():
    """
    Render the complete
    difficulty evaluation section.
    """

    st.header("Performance by Difficulty")

    df, best_models = prepare_difficulty_summary()

    render_difficulty_cards(
        best_models,
    )

    st.divider()

    render_summary_table(
        df,
    )

    st.divider()

    render_difficulty_insights(
        df,
    )
