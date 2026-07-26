from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent

STYLE_DIR = BASE_DIR / "assets" / "styles"


def load_css(filename: str):

    css_file = STYLE_DIR / filename

    with open(css_file, encoding="utf-8") as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )


def load_theme():

    load_css("theme.css")

    load_css("animations.css")