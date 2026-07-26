import time

import streamlit as st


def render_typewriter(
    text: str,
    *,
    delay: float = 0.001,
) -> None:
    """
    Render text with a typewriter animation.

    Parameters
    ----------
    text:
        Text to render.

    delay:
        Delay (seconds) between characters.
    """

    placeholder = st.empty()

    rendered = ""

    for char in text:

        rendered += char

        placeholder.markdown(rendered)

        time.sleep(delay)

    # Final render to ensure Markdown is parsed correctly.
    placeholder.markdown(text)
