from __future__ import annotations

import pandas as pd
import streamlit as st

# pyrefly: ignore [missing-import]
from utils.data_loader import (
    EvaluationDataError,
    build_case_study_rankings,
    format_configuration,
    get_case_study_queries,
    get_case_study_query_metadata,
)

DISPLAY_COLUMNS = [
    "rank",
    "title",
    "genres",
    "evaluation_score",
    "relevance",
    "semantic_similarity",
    "cross_encoder_score",
]


def _format_case_study_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df[DISPLAY_COLUMNS].copy()
    display["evaluation_score"] = display["evaluation_score"].round(3)
    display["semantic_similarity"] = display["semantic_similarity"].round(3)
    display["cross_encoder_score"] = display["cross_encoder_score"].round(3)
    return display


def _render_ranked_table(title: str, df: pd.DataFrame):
    st.markdown(f"**{title}**")
    st.dataframe(
        _format_case_study_table(df),
        hide_index=True,
        use_container_width=True,
    )


def render_case_study_viewer(
    per_query_df: pd.DataFrame,
    scored_predictions: pd.DataFrame | None = None,
):
    st.subheader("Case Studies")

    if scored_predictions is None or scored_predictions.empty:
        st.info(
            "scored_predictions.csv is not available with the expected schema, so this section falls back to per-query summary mode."
        )

        query_options = (
            per_query_df[["query_id", "query", "category", "difficulty"]]
            .drop_duplicates()
            .sort_values(["category", "difficulty", "query_id"])
        )
        selected_query_id = st.selectbox(
            "Select query",
            query_options["query_id"].tolist(),
        )

        query_row = query_options[query_options["query_id"] == selected_query_id].iloc[
            0
        ]
        st.write(
            f"**{query_row['query_id']}** · {query_row['query']} · {query_row['category']} · {query_row['difficulty']}"
        )

        query_df = per_query_df[per_query_df["query_id"] == selected_query_id].copy()
        st.dataframe(query_df, hide_index=True, use_container_width=True)
        return

    query_options = get_case_study_queries(scored_predictions)
    labels = [
        f"{row.query_id} · {row.query} · {row.category} · {row.difficulty}"
        for row in query_options.itertuples(index=False)
    ]

    selected_label = st.selectbox("Select query", labels)
    selected_row = query_options.iloc[labels.index(selected_label)]
    metadata = get_case_study_query_metadata(
        scored_predictions,
        selected_row["query_id"],
    )

    st.markdown(f"""
**{metadata['query_id']}**

{metadata['query']}

Category: **{metadata['category']}** · Difficulty: **{metadata['difficulty']}**
""")

    try:
        rankings = build_case_study_rankings(
            scored_predictions,
            metadata["query_id"],
        )
    except EvaluationDataError as exc:
        st.error(str(exc))
        return

    left_label = format_configuration("faiss_only")
    right_label = format_configuration("hybrid_with_ce")

    col_left, col_right = st.columns(2)

    with col_left:
        if "faiss_only" in rankings:
            _render_ranked_table(left_label, rankings["faiss_only"])
        else:
            st.warning("FAISS-only ranking not available for this query.")

    with col_right:
        if "hybrid_with_ce" in rankings:
            _render_ranked_table(right_label, rankings["hybrid_with_ce"])
        else:
            st.warning("Hybrid + Cross Encoder ranking not available for this query.")

    if "faiss_only" in rankings and "hybrid_with_ce" in rankings:
        merged = rankings["faiss_only"][
            ["rank", "title", "evaluation_score", "relevance"]
        ].merge(
            rankings["hybrid_with_ce"][
                ["rank", "title", "evaluation_score", "relevance"]
            ],
            on="rank",
            how="outer",
            suffixes=("_faiss", "_hybrid"),
        )

        st.markdown("**Rank-by-rank comparison**")
        st.dataframe(merged, hide_index=True, use_container_width=True)
