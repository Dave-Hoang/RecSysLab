import hashlib

import streamlit as st

from components.typewriter import render_typewriter


def render_explanation(
    explanation: str | None,
) -> None:
    """
    Render AI-generated explanation.

    The typewriter animation is played only once for
    each unique explanation.
    """

    if not explanation:
        return

    st.markdown("### 💡 AI Explanation")

    explanation_key = "animated_" + hashlib.md5(explanation.encode("utf-8")).hexdigest()

    if st.session_state.get(explanation_key):

        st.markdown(explanation)

        return

    render_typewriter(explanation)

    st.session_state[explanation_key] = True
