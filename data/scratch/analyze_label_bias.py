"""
Analyze potential bias in LLM-generated labels vs existing labels.

Compares relevance score distributions between:
1. Existing labels (human or older)
2. New LLM-generated labels (from auto_label_missing.py)
"""
import pandas as pd
import numpy as np
from scipy import stats

def analyze_bias():
    labels = pd.read_csv('d:/Project/RecSysLab/data/evaluation/labels.csv')
    agentic_preds = pd.read_csv('d:/Project/RecSysLab/data/evaluation/agentic_predictions.csv')

    # Identify which labels are for Agentic predictions
    agentic_labels = pd.merge(
        agentic_preds[['query_id', 'movieId']],
        labels,
        on=['query_id', 'movieId'],
        how='inner'
    )

    # Split into two groups:
    # Group A: Labels with detailed reasoning (LLM-generated, likely new)
    # Group B: Labels with empty or short notes (existing/human)

    llm_generated = labels[labels['notes'].str.len() > 50]  # Long reasoning = LLM
    existing = labels[labels['notes'].str.len() <= 50]  # Short/empty = existing

    print("=" * 70)
    print("LABEL BIAS ANALYSIS")
    print("=" * 70)

    print(f"\nTotal labels: {len(labels)}")
    print(f"LLM-generated labels (notes > 50 chars): {len(llm_generated)}")
    print(f"Existing labels (notes <= 50 chars): {len(existing)}")

    # Distribution comparison
    print("\n" + "=" * 70)
    print("RELEVANCE SCORE DISTRIBUTION")
    print("=" * 70)

    print("\nLLM-Generated Labels:")
    print(llm_generated['relevance'].value_counts().sort_index())
    llm_mean = llm_generated['relevance'].mean()
    print(f"Mean: {llm_mean:.3f}")

    print("\nExisting Labels:")
    print(existing['relevance'].value_counts().sort_index())
    existing_mean = existing['relevance'].mean()
    print(f"Mean: {existing_mean:.3f}")

    # Statistical test
    print("\n" + "=" * 70)
    print("STATISTICAL TEST (Mann-Whitney U)")
    print("=" * 70)

    statistic, pvalue = stats.mannwhitneyu(
        llm_generated['relevance'],
        existing['relevance'],
        alternative='greater'  # Test if LLM-generated scores are higher
    )

    print(f"H0: LLM-generated labels have same/lower relevance scores")
    print(f"H1: LLM-generated labels have HIGHER relevance scores")
    print(f"Test statistic: {statistic:.2f}")
    print(f"P-value: {pvalue:.4f}")

    if pvalue < 0.05:
        print("\n🚨 BIAS DETECTED: LLM-generated labels have significantly higher scores (p < 0.05)")
        print(f"   Mean difference: +{llm_mean - existing_mean:.3f} points")
    else:
        print("\n✅ No significant bias detected (p >= 0.05)")

    # Category-level analysis
    print("\n" + "=" * 70)
    print("CATEGORY-LEVEL ANALYSIS")
    print("=" * 70)

    # Focus on categories with many LLM-generated labels
    categories_with_llm = llm_generated['category'].value_counts()
    print("\nCategories with LLM-generated labels:")
    print(categories_with_llm)

    # For each category, compare mean relevance
    print("\n--- Mean Relevance by Category ---")
    for cat in categories_with_llm.index:
        llm_cat = llm_generated[llm_generated['category'] == cat]['relevance'].mean()
        existing_cat = existing[existing['category'] == cat]['relevance'].mean()
        diff = llm_cat - existing_cat

        print(f"{cat:20} | LLM: {llm_cat:.3f} | Existing: {existing_cat:.3f} | Diff: {diff:+.3f}")

    # Agentic-specific analysis
    print("\n" + "=" * 70)
    print("AGENTIC MODE LABEL ANALYSIS")
    print("=" * 70)

    agentic_llm = pd.merge(
        agentic_preds[['query_id', 'movieId']],
        llm_generated,
        on=['query_id', 'movieId'],
        how='inner'
    )

    agentic_existing = pd.merge(
        agentic_preds[['query_id', 'movieId']],
        existing,
        on=['query_id', 'movieId'],
        how='inner'
    )

    print(f"\nAgentic predictions with LLM-generated labels: {len(agentic_llm)}")
    print(f"Agentic predictions with existing labels: {len(agentic_existing)}")

    if len(agentic_llm) > 0:
        print(f"\nLLM-generated labels for Agentic:")
        print(agentic_llm['relevance'].value_counts().sort_index())
        print(f"Mean: {agentic_llm['relevance'].mean():.3f}")

    if len(agentic_existing) > 0:
        print(f"\nExisting labels for Agentic:")
        print(agentic_existing['relevance'].value_counts().sort_index())
        print(f"Mean: {agentic_existing['relevance'].mean():.3f}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if pvalue < 0.05:
        print("""
🚨 BIAS CONFIRMED - Results are NOT reliable for comparison.

Recommended actions:
1. Remove auto-generated labels from evaluation
2. Use only human-labeled or pre-existing labels for NDCG calculation
3. If keeping LLM labels, run separate analysis:
   - NDCG with only existing labels
   - NDCG with all labels (mark as "potentially biased")
4. Consider calibrating LLM judge with few-shot examples from existing labels
        """)
    else:
        print("""
✅ No significant bias detected, but still recommend:
1. Manual review of 20-30 random LLM-generated labels
2. Check consistency with existing label standards
3. Monitor category-specific biases
        """)

if __name__ == "__main__":
    analyze_bias()
