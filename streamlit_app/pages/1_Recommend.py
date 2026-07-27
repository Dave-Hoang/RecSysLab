import streamlit as st

from utils.theme import load_theme
from services.api_client import api_client
from components.movie_card import render_movie_card
from components.latency import render_timings

load_theme()

st.title("🎬 Movie Recommendation")

st.write("Search movies using the Hybrid Semantic Recommendation System.")

# ===========================
# Search Form
# ===========================


def clear_animation_state() -> None:
    keys = [key for key in st.session_state if key.startswith("animated_")]

    for key in keys:
        del st.session_state[key]


with st.form("recommend_form"):

    query = st.text_input(
        "Search Query",
        placeholder="psychological sci-fi movies",
    )

    mode = st.radio(
        "Recommendation Mode",
        ["quality", "fast"],
        horizontal=True,
    )

    top_k = st.slider(
        "Top K",
        min_value=1,
        max_value=20,
        value=5,
    )

    include_explanation = st.checkbox(
        "Generate LLM Explanation",
        value=True,
    )

    submitted = st.form_submit_button("🔍 Recommend")

# ===========================
# Recommendation Pipeline
# ===========================

if submitted:

    if not query.strip():

        st.warning("Please enter a search query.")

    else:

        try:
            clear_animation_state()

            with st.spinner("Generating recommendations..."):
                response = api_client.recommend(
                    query=query,
                    mode=mode,
                    top_k=top_k,
                    include_explanation=include_explanation,
                )

            recommendations = response.get(
                "recommendations",
                [],
            )

            st.subheader("Recommended Movies")

            if not recommendations:

                st.info("No recommendations found.")

            else:

                for idx, movie in enumerate(
                    recommendations,
                    start=1,
                ):

                    render_movie_card(
                        movie,
                        idx,
                    )

            timings = response.get("timings")

            if timings:

                render_timings(timings)

        except Exception as e:

            st.error("Recommendation failed")

            st.exception(e)
