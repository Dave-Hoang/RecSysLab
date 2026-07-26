import streamlit as st


def render_badges(genres: str | None) -> None:
    """
    Render movie genres as badge pills.

    Example
    -------
    Input:
        "Action|Adventure|Sci-Fi"

    Output:
        [ Action ] [ Adventure ] [ Sci-Fi ]
    """

    if not genres:
        return

    genre_list = [
        genre.strip()
        for genre in genres.split("|")
        if genre.strip()
    ]

    badges = ""

    for genre in genre_list:
        badges += (
            f'<span class="genre-badge">{genre}</span>'
        )

    st.markdown(
        badges,
        unsafe_allow_html=True,
    )