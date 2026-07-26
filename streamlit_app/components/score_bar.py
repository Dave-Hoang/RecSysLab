import streamlit as st


def render_score_bar(
    label: str,
    value: float | None,
) -> None:
    """
    Render normalized score (0-1)
    as a gradient progress bar.
    """

    if value is None:
        return

    value = max(0.0, min(float(value), 1.0))

    percentage = value * 100

    st.markdown(
        f"""
<div class="score-container">

<div class="score-label">

<span>{label}</span>

<span>{value:.3f}</span>

</div>

<div class="score-track">

<div
class="score-fill"
style="width:{percentage:.1f}%">
</div>

</div>

</div>
""",
        unsafe_allow_html=True,
    )