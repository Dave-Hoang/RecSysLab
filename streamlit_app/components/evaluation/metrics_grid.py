from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.data_loader import (
    DEFAULT_CHALLENGER_CONFIGURATION,
    build_case_study_candidates,
    format_configuration,
)


def _best_row(df: pd.DataFrame, configuration: str, metric: str) -> pd.Series:
    subset = df[df["configuration"] == configuration]

    if subset.empty:
        raise ValueError(f"Missing configuration {configuration}")

    return subset.sort_values(metric, ascending=False).iloc[0]


def render_metrics_grid(
    overall_df: pd.DataFrame,
    category_df: pd.DataFrame,
    difficulty_df: pd.DataFrame,
    per_query_df: pd.DataFrame,
):
    st.subheader("Key Results")

    best_category = _best_row(
        category_df,
        DEFAULT_CHALLENGER_CONFIGURATION,
        "ndcg_at_5_mean",
    )

    hardest_bucket = (
        difficulty_df[
            difficulty_df["configuration"] == DEFAULT_CHALLENGER_CONFIGURATION
        ]
        .sort_values("ndcg_at_5_mean", ascending=True)
        .iloc[0]
    )

    strongest_query = build_case_study_candidates(per_query_df).iloc[0]

    category_winners = category_df.groupby("category")["ndcg_at_5_mean"].idxmax()
    winning_configurations = category_df.loc[
        category_winners, "configuration"
    ].value_counts()
    category_win_count = (
        int(winning_configurations.max()) if not winning_configurations.empty else 0
    )

    top_lift = strongest_query["delta_ndcg_at_5"] * 100
    overall_lift = (
        overall_df.loc[
            overall_df["configuration"] == DEFAULT_CHALLENGER_CONFIGURATION,
            "ndcg_at_5_mean",
        ].iloc[0]
        - overall_df.loc[
            overall_df["configuration"] == "faiss_only", "ndcg_at_5_mean"
        ].iloc[0]
    ) * 100

    top_row = st.columns(3)
    bottom_row = st.columns(3)

    with top_row[0]:
        st.metric(
            "Overall NDCG Lift",
            f"+{overall_lift:.1f}%",
            format_configuration(DEFAULT_CHALLENGER_CONFIGURATION),
        )

    with top_row[1]:
        st.metric(
            "Best Category",
            best_category["category"].replace("_", " ").title(),
            f"NDCG@5 {best_category['ndcg_at_5_mean']:.4f}",
        )

    with top_row[2]:
        st.metric(
            "Hardest Scenario",
            str(hardest_bucket["difficulty"]).title(),
            f"NDCG@5 {hardest_bucket['ndcg_at_5_mean']:.4f}",
        )

    with bottom_row[0]:
        irrelevant_drop = (
            overall_df.loc[
                overall_df["configuration"] == "faiss_only", "irrelevant_at_5_mean"
            ].iloc[0]
            - overall_df.loc[
                overall_df["configuration"] == DEFAULT_CHALLENGER_CONFIGURATION,
                "irrelevant_at_5_mean",
            ].iloc[0]
        )
        st.metric(
            "Noise Reduction",
            f"{irrelevant_drop:.3f}",
            "fewer irrelevant items@5",
        )

    with bottom_row[1]:
        st.metric(
            "Strongest Query Lift",
            strongest_query["query_id"],
            f"+{top_lift:.1f}% NDCG@5",
        )

    with bottom_row[2]:
        st.empty()

    st.caption(
        f"{format_configuration(DEFAULT_CHALLENGER_CONFIGURATION)} wins the most category buckets in this snapshot: {category_win_count}."
    )
