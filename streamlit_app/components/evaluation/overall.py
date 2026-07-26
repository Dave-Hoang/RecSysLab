import streamlit as st

from components.evaluation.utils import (
    format_configuration,
    load_overall_metrics,
)

DISPLAY_COLUMNS = {
    "configuration": "Method",
    "precision_at_5_mean": "Precision@5",
    "ndcg_at_5_mean": "NDCG@5",
    "mrr_at_5_mean": "MRR@5",
    "mean_relevance_at_5_mean": "Mean Relevance",
}


def render_overall_metrics():

    st.header("Overall Performance")

    df = load_overall_metrics()

    best = df.sort_values(
        "ndcg_at_5_mean",
        ascending=False,
    ).iloc[0]

    st.success(f"""
### 🏆 Best Overall Pipeline

**{format_configuration(best["configuration"])}**

- Precision@5 : **{best["precision_at_5_mean"]:.3f}**
- NDCG@5 : **{best["ndcg_at_5_mean"]:.3f}**
- MRR@5 : **{best["mrr_at_5_mean"]:.3f}**
""")

    st.subheader("Evaluation Summary")

    summary = df[list(DISPLAY_COLUMNS.keys())].rename(columns=DISPLAY_COLUMNS).copy()

    summary["Method"] = summary["Method"].apply(format_configuration)

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Key Findings")

    precision_best = df.loc[df["precision_at_5_mean"].idxmax()]

    ndcg_best = df.loc[df["ndcg_at_5_mean"].idxmax()]

    mrr_best = df.loc[df["mrr_at_5_mean"].idxmax()]

    relevance_gain = (
        (
            best["mean_relevance_at_5_mean"]
            - df.loc[
                df["configuration"] == "faiss_only",
                "mean_relevance_at_5_mean",
            ].iloc[0]
        )
        / df.loc[
            df["configuration"] == "faiss_only",
            "mean_relevance_at_5_mean",
        ].iloc[0]
        * 100
    )

    st.markdown(f"""
- **Highest Precision@5:** {format_configuration(precision_best["configuration"])}
- **Highest NDCG@5:** {format_configuration(ndcg_best["configuration"])}
- **Highest MRR@5:** {format_configuration(mrr_best["configuration"])}
- **Mean relevance improved by {relevance_gain:.1f}% over FAISS.**
""")
