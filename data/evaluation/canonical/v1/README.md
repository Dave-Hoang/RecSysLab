# Canonical Evaluation v1

This directory contains the canonical benchmark artifacts for the Agentic Movie Recommendation System.

## Evaluation Protocol
- **Queries**: 30 diverse test queries across 6 categories (genre, emotion, negative constraint, multi-condition, similar movie, natural language).
- **Labels**: Dense human judgments (v2) covering 100% of the Top-5 predictions from both systems. Scale: 0 (Irrelevant), 1 (Relevant), 2 (Highly Relevant).
- **Metrics**: nDCG@5 and Precision@5 calculated with strict mode (any missing label results in NaN).

## Configurations
- **Agentic Pipeline (Production)**: Uses LangGraph for adaptive query expansion, intent routing, and quality gates. Ranking weights: CE=0.55 / Semantic=0.25 / Popularity=0.10 / Rule=0.10.
- **Classic Pipeline (Ablation Baseline)**: Single-pass retrieval and ranking. Ranking weights: CE=0.50 / Semantic=0.25 / Popularity=0.15 / Rule=0.10.

> **Historical Note on Stale Metrics (0.9264)**: Early beta tests reported an Agentic nDCG@5 of 0.9264. That result is invalid and superseded by this v1 benchmark because it was calculated on a 25-query subset (pre-label merge), lacked complete human labels, and suffered from a rank-compaction bug during the dataframe inner-join. The canonical metrics in this directory (Agentic: 0.843, Classic: 0.809) represent the true, verified performance.
