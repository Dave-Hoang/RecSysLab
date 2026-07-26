from __future__ import annotations

import streamlit as st

from utils.data_loader import (
    build_overall_headlines,
    compare_to_baseline,
    format_configuration,
)


def render_hero_card(overall_df):
    headline = build_overall_headlines(overall_df)
    comparison = compare_to_baseline(
        overall_df,
        metric_columns=(
            "precision_at_5_mean",
            "ndcg_at_5_mean",
            "mrr_at_5_mean",
            "irrelevant_at_5_mean",
            "mean_relevance_at_5_mean",
        ),
    )

    st.subheader("Best Pipeline Snapshot")

    with st.container(border=True):
        st.markdown(f"""
### 🏆 {format_configuration(headline['best_configuration'])}

The strongest overall pipeline across the offline evaluation set.
""")

        top_row = st.columns(3)
        bottom_row = st.columns(3)

        with top_row[0]:
            st.metric(
                "Precision@5",
                f"{headline['best_precision_at_5_mean']:.3f}",
                f"+{comparison['delta_precision_at_5_mean']:.3f} vs FAISS",
            )

        with top_row[1]:
            st.metric(
                "NDCG@5",
                f"{headline['best_ndcg_at_5_mean']:.3f}",
                f"+{comparison['delta_ndcg_at_5_mean']:.3f} vs FAISS",
            )

        with top_row[2]:
            st.metric(
                "MRR@5",
                f"{headline['best_mrr_at_5_mean']:.3f}",
                f"+{comparison['delta_mrr_at_5_mean']:.3f} vs FAISS",
            )

        with bottom_row[0]:
            st.metric(
                "Irrelevant@5",
                f"{comparison['challenger_irrelevant_at_5_mean']:.3f}",
                f"{comparison['delta_irrelevant_at_5_mean']:.3f} vs FAISS",
            )

        with bottom_row[1]:
            st.metric(
                "Mean Relevance@5",
                f"{headline['best_mean_relevance_at_5_mean']:.3f}",
                f"+{comparison['delta_mean_relevance_at_5_mean']:.3f} vs FAISS",
            )

        with bottom_row[2]:
            st.metric(
                "Query Coverage",
                int(overall_df.iloc[0]["query_count"]),
                "30 labeled queries",
            )

        st.caption(
            "FAISS is the baseline; all improvement numbers are measured against that pipeline."
        )
