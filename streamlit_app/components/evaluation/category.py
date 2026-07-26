import pandas as pd
import streamlit as st

from components.evaluation.utils import (
    format_configuration,
    load_category_metrics,
)

CATEGORY_ORDER = [
    "emotion_theme",
    "genre",
    "multi_condition",
    "natural_language",
    "negative_constraint",
    "similar_movie",
]


CATEGORY_NAMES = {
    "emotion_theme": "Emotion Theme",
    "genre": "Genre",
    "multi_condition": "Multi-condition",
    "natural_language": "Natural Language",
    "negative_constraint": "Negative Constraint",
    "similar_movie": "Similar Movie",
}


SUMMARY_COLUMNS = {
    "configuration": "Method",
    "precision_at_5_mean": "Precision@5",
    "ndcg_at_5_mean": "NDCG@5",
    "mrr_at_5_mean": "MRR@5",
    "mean_relevance_at_5_mean": "Mean Relevance",
}


def get_best_configuration(
    category_df: pd.DataFrame,
    metric: str = "ndcg_at_5_mean",
) -> pd.Series:
    """
    Return the best configuration for one category.
    """

    return category_df.sort_values(metric, ascending=False).iloc[0]


def prettify_category(category: str) -> str:
    """
    Convert category id into display name.
    """

    return CATEGORY_NAMES.get(category, category)


def build_summary_table(
    category_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build display table for one category.
    """

    summary = (
        category_df[list(SUMMARY_COLUMNS.keys())].rename(columns=SUMMARY_COLUMNS).copy()
    )

    summary["Method"] = summary["Method"].apply(format_configuration)

    return summary


def prepare_category_summary() -> tuple[pd.DataFrame, dict]:
    """
    Returns

    raw dataframe

    best model information
    """

    df = load_category_metrics()

    best_models = {}

    for category in CATEGORY_ORDER:

        category_df = df[df["category"] == category].copy()

        best_models[category] = get_best_configuration(category_df)

    return df, best_models


def render_category_cards(
    best_models: dict,
):
    """
    Render one metric card for each category.
    """

    st.subheader("Best Model by Category")

    columns = st.columns(3)

    for index, category in enumerate(CATEGORY_ORDER):

        column = columns[index % 3]

        best = best_models[category]

        with column:

            st.metric(
                label=prettify_category(category),
                value=format_configuration(best["configuration"]),
                delta=f'NDCG@5 {best["ndcg_at_5_mean"]:.3f}',
            )


def render_summary_table(
    df: pd.DataFrame,
):
    """
    Render category comparison table.
    """

    st.subheader("Category Summary")

    rows = []

    for category in CATEGORY_ORDER:

        category_df = df[df["category"] == category]

        best = get_best_configuration(category_df)

        rows.append(
            {
                "Category": prettify_category(category),
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


def render_category_insights(
    df: pd.DataFrame,
):
    """
    Generate simple observations from category metrics.
    """

    st.subheader("Observations")

    winners = df.groupby("configuration")["ndcg_at_5_mean"].idxmax()

    winner_count = df.loc[winners].groupby("configuration").size()

    overall_best = winner_count.idxmax()

    overall_count = winner_count.max()

    multi_condition = (
        df[df["category"] == "multi_condition"]
        .sort_values(
            "ndcg_at_5_mean",
            ascending=False,
        )
        .iloc[0]
    )

    natural_language = (
        df[df["category"] == "natural_language"]
        .sort_values(
            "ndcg_at_5_mean",
            ascending=False,
        )
        .iloc[0]
    )

    st.markdown(f"""
- **{format_configuration(overall_best)}** achieved the best NDCG@5 in **{overall_count}** query categories.

- The strongest performance on **Multi-condition** queries was achieved by **{format_configuration(multi_condition["configuration"])}**.

- The strongest performance on **Natural Language** queries was achieved by **{format_configuration(natural_language["configuration"])}**.

- Query category has a noticeable impact on recommendation quality, indicating that different retrieval pipelines excel under different semantic constraints.
""")


def render_category_metrics():
    """
    Render category evaluation section.
    """

    st.header("Performance by Query Category")

    df, best_models = prepare_category_summary()

    render_category_cards(
        best_models,
    )

    st.divider()

    render_summary_table(
        df,
    )

    st.divider()

    render_category_insights(
        df,
    )
