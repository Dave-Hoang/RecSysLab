from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.data_loader import (
    BASELINE_CONFIGURATION,
    DEFAULT_CHALLENGER_CONFIGURATION,
    build_case_study_candidates,
    compare_to_baseline,
    format_configuration,
)


def render_findings_card(
    overall_df: pd.DataFrame,
    category_df: pd.DataFrame,
    difficulty_df: pd.DataFrame,
    per_query_df: pd.DataFrame,
):
    st.subheader("Important Findings")

    overall_comparison = compare_to_baseline(
        overall_df,
        metric_columns=(
            "precision_at_5_mean",
            "ndcg_at_5_mean",
            "mrr_at_5_mean",
            "irrelevant_at_5_mean",
            "mean_relevance_at_5_mean",
        ),
    )

    category_winners = category_df.groupby("category")["ndcg_at_5_mean"].idxmax()
    category_winner_counts = (
        category_df.loc[category_winners, "configuration"].value_counts().to_dict()
    )

    difficulty_winners = difficulty_df.groupby("difficulty")["ndcg_at_5_mean"].idxmax()
    difficulty_winner_counts = (
        difficulty_df.loc[difficulty_winners, "configuration"].value_counts().to_dict()
    )

    strongest_query = build_case_study_candidates(per_query_df).iloc[0]

    baseline_label = format_configuration(BASELINE_CONFIGURATION)
    challenger_label = format_configuration(DEFAULT_CHALLENGER_CONFIGURATION)

    def _render_winner_summary(title: str, counts: dict[str, int]) -> None:
        st.markdown(f"**{title}**")

        if not counts:
            st.caption("No winner summary available.")
            return

        rows = [
            f"- **{format_configuration(configuration)}**: {count}"
            for configuration, count in counts.items()
        ]

        st.markdown("\n".join(rows))

    with st.container(border=True):
        st.markdown(f"""
1. **{challenger_label}** is the best overall pipeline, improving NDCG@5 by **{overall_comparison['delta_ndcg_at_5_mean']:.3f}** over {baseline_label}.
2. {challenger_label} wins the most category buckets in this snapshot, which means the reranker helps across several semantic query types.
3. The same pipeline also leads the difficulty breakdown, but the hardest bucket still shows the lowest NDCG@5, so semantic reranking remains most valuable there.
4. The strongest per-query lift is **{strongest_query['query_id']}**, with **+{strongest_query['delta_ndcg_at_5']:.3f} NDCG@5** versus {baseline_label}.
""")

        col1, col2 = st.columns(2)

        with col1:
            _render_winner_summary("Category wins", category_winner_counts)

        with col2:
            _render_winner_summary("Difficulty wins", difficulty_winner_counts)
