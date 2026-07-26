import streamlit as st

from utils.bootstrap import ensure_project_root

ensure_project_root()

st.set_page_config(
    page_title="RecSysLab",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.1rem;
    padding-left: 1.1rem;
    padding-right: 1.1rem;
    max-width: 100%;
}

@media (max-width: 900px) {
    .block-container {
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

from components.evaluation import (
    render_case_study_viewer,
    render_findings_card,
    render_hero_card,
    render_metrics_grid,
)

from utils.data_loader import (
    EvaluationDataError,
    build_evaluation_bundle,
    build_overall_headlines,
    compare_to_baseline,
    format_configuration,
)

from utils.theme import load_theme

load_theme()

try:
    bundle = build_evaluation_bundle()
except (FileNotFoundError, EvaluationDataError) as exc:
    st.error(str(exc))
    st.stop()

headline = build_overall_headlines(bundle.overall)
overall_comparison = compare_to_baseline(
    bundle.overall,
    metric_columns=(
        "precision_at_5_mean",
        "ndcg_at_5_mean",
        "mrr_at_5_mean",
        "irrelevant_at_5_mean",
        "mean_relevance_at_5_mean",
    ),
)

st.title("📊 System Evaluation")

st.markdown(
    """
<div class="evaluation-panel evaluation-lead evaluation-story">
<p>This page presents the offline evaluation results of the Hybrid Semantic Movie Recommendation System.</p>
<p>The evaluation compares four retrieval pipelines across multiple recommendation metrics, query categories, and difficulty levels.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="evaluation-panel evaluation-story">
<h3>Story Flow</h3>
<ol>
<li>Start with the best pipeline snapshot.</li>
<li>Read the headline results and technical findings.</li>
<li>Inspect one query-level case study.</li>
</ol>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

render_hero_card(bundle.overall)

st.divider()

render_metrics_grid(
    bundle.overall,
    bundle.category,
    bundle.difficulty,
    bundle.per_query,
)

st.divider()

render_findings_card(
    bundle.overall,
    bundle.category,
    bundle.difficulty,
    bundle.per_query,
)

st.divider()

render_case_study_viewer(
    bundle.per_query,
    bundle.scored_predictions,
)

st.divider()

with st.container(border=True):
    st.subheader("Overall Conclusion")

    st.success(
        f"Final takeaway: {format_configuration(headline['best_configuration'])} is the strongest end-to-end pipeline for this benchmark."
    )

    conclusion_columns = st.columns(3)

    with conclusion_columns[0]:
        st.metric(
            "NDCG@5 Gain",
            f"+{overall_comparison['delta_ndcg_at_5_mean']:.3f}",
            "vs FAISS",
        )

    with conclusion_columns[1]:
        st.metric(
            "Precision@5 Gain",
            f"+{overall_comparison['delta_precision_at_5_mean']:.3f}",
            "vs FAISS",
        )

    with conclusion_columns[2]:
        st.metric(
            "Lower Noise",
            f"{-overall_comparison['delta_irrelevant_at_5_mean']:.3f}",
            "irrelevant@5 reduction",
        )

    st.markdown(
        """
<div class="evaluation-caption">
The ranking story is consistent across the dashboard: Hybrid + Cross Encoder gives the clearest overall lift, especially on NDCG@5.
</div>
""",
        unsafe_allow_html=True,
    )
