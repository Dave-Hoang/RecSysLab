import streamlit as st


def render_timings(timings: dict):

    if not timings:
        return

    st.subheader("⚡ Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Ranking",
            f"{timings.get('ranking_seconds', 0):.3f}s"
        )

    with col2:
        st.metric(
            "Generation",
            f"{timings.get('generation_seconds', 0):.3f}s"
        )

    with col3:
        st.metric(
            "Total",
            f"{timings.get('total_seconds', 0):.3f}s"
        )