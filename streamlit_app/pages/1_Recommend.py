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

    pipeline_mode = st.radio(
        "Pipeline Mode",
        ["Classic (Rule-based)", "🤖 Agentic Mode (LangGraph)"],
        horizontal=True,
    )

    mode = st.radio(
        "Ranking Mode (Classic Only)",
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

            is_agentic = "Agentic" in pipeline_mode

            with st.spinner("Generating recommendations..."):
                if is_agentic:
                    response = api_client.agentic_recommend(
                        query=query,
                        top_k=top_k,
                        include_explanation=include_explanation,
                    )
                else:
                    response = api_client.recommend(
                        query=query,
                        mode=mode,
                        top_k=top_k,
                        include_explanation=include_explanation,
                    )

            if is_agentic:
                st.subheader("🤖 Agentic Trace")
                st.write(f"**Intent:** `{response.get('intent', 'N/A')}`")
                
                expanded_query = response.get("expanded_query")
                if expanded_query and expanded_query != query:
                    st.write(f"**Expanded Query:** {expanded_query}")
                
                confidence = response.get("confidence_level")
                if confidence:
                    st.write(f"**Confidence Level:** `{confidence}`")
                
                execution_path = response.get("execution_path", [])
                if execution_path:
                    path_str = " → ".join(execution_path)
                    st.write(f"**Execution Path:** {path_str}")

                direct_response = response.get("direct_response")
                if direct_response:
                    st.info(f"💬 {direct_response}")

            recommendations = response.get(
                "recommendations",
                [],
            )

            if not response.get("direct_response"):
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
