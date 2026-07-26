import streamlit as st

from components.badge import render_badges
from components.score_bar import render_score_bar
from components.explanation import render_explanation

def _format_rating(value):
    if value is None:
        return "-"
    return f"{int(value):,}"


def _format_count(value):
    if value is None:
        return "-"
    return f"{value:,}"


def _format_score(value):
    if value is None:
        return "-"
    return f"{value:.3f}"


def render_movie_card(movie: dict, rank: int):
    """
    Render a recommendation result card.
    """

    title = movie.get("title", "Unknown Movie")

    genres = movie.get("genres", "")

    rating_mean = movie.get("rating_mean")
    rating_count = movie.get("rating_count")

    semantic = movie.get("semantic_similarity")
    cross_encoder = movie.get("cross_encoder_score")
    final_score = movie.get("final_score")

    retrieval_rank = movie.get("retrieval_rank")
    result_rank = movie.get("result_rank")
    final_rank = movie.get("final_rank")

    popularity = movie.get("popularity_score")
    rule_score = movie.get("rule_score")

    explanation = movie.get("explanation")

    metadata = movie.get("page_content", "")

    st.markdown("---")

    st.subheader(f"{rank}. 🎬 {title}")

    render_badges(genres)

    st.markdown(
        f"""
⭐ **{_format_rating(rating_mean)}**
&nbsp;&nbsp;&nbsp;&nbsp;
👥 **{_format_count(rating_count)} ratings**
""",
        unsafe_allow_html=True,
    )

    render_score_bar(
        "Final Score",
        final_score,
    )

    render_score_bar(
        "Semantic Similarity",
        semantic,
    )

    render_score_bar(
        "Cross Encoder Score",
        cross_encoder,
    )

    with st.expander("Technical Details"):

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**Retrieval Rank:** {retrieval_rank}"
            )

            st.write(
                f"**Result Rank:** {result_rank}"
            )

            st.write(
                f"**Final Rank:** {final_rank}"
            )

        with col2:

            st.write(
                f"**Popularity Score:** {_format_score(popularity)}"
            )

            st.write(
                f"**Rule Score:** {_format_score(rule_score)}"
            )

    render_explanation(explanation)

    with st.expander("Embedding Source Text"):

        st.text(metadata)