"""
Recompute NDCG using ONLY existing (non-LLM-generated) labels.

This removes the 54 LLM-generated labels with inflated scores (mean 1.76)
and keeps only the 291 existing labels (mean ~0.45-1.5 depending on filtering).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import EVALUATION_DIR

def compute_ndcg_unbiased():
    # Load data
    labels = pd.read_csv(EVALUATION_DIR / 'labels.csv')
    agentic_preds = pd.read_csv(EVALUATION_DIR / 'agentic_predictions.csv')
    classic_preds = pd.read_csv(EVALUATION_DIR / 'predictions.csv')

    # Filter: Keep ONLY existing labels (notes <= 50 chars)
    # LLM-generated labels have long reasoning (> 50 chars)
    existing_labels = labels[labels['notes'].str.len() <= 50].copy()

    print("=" * 70)
    print("UNBIASED NDCG EVALUATION (EXISTING LABELS ONLY)")
    print("=" * 70)

    print(f"\nTotal labels: {len(labels)}")
    print(f"Existing labels (used): {len(existing_labels)}")
    print(f"LLM-generated labels (excluded): {len(labels) - len(existing_labels)}")

    print(f"\nExisting labels mean relevance: {existing_labels['relevance'].mean():.3f}")
    print("Relevance distribution:")
    print(existing_labels['relevance'].value_counts().sort_index())

    # Helper function: NDCG@k
    def ndcg_at_k(relevances, k=5):
        """Compute NDCG@k for a single query."""
        relevances = np.array(relevances[:k])
        if len(relevances) == 0:
            return 0.0

        # DCG
        dcg = relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, len(relevances) + 1)))

        # IDCG (ideal)
        ideal_relevances = np.sort(relevances)[::-1]
        idcg = ideal_relevances[0] + np.sum(ideal_relevances[1:] / np.log2(np.arange(2, len(ideal_relevances) + 1)))

        return dcg / idcg if idcg > 0 else 0.0

    # Compute NDCG for each configuration
    results = []

    # Classic configurations
    classic_configs = classic_preds['configuration'].unique()
    for config in classic_configs:
        config_preds = classic_preds[classic_preds['configuration'] == config]

        # Merge with existing labels
        merged = pd.merge(
            config_preds,
            existing_labels[['query_id', 'movieId', 'relevance', 'category']],
            on=['query_id', 'movieId'],
            how='inner'
        )

        if len(merged) == 0:
            continue

        # Compute NDCG per query
        ndcgs = []
        for qid in merged['query_id'].unique():
            query_data = merged[merged['query_id'] == qid].sort_values('rank')
            relevances = query_data['relevance'].tolist()
            ndcgs.append(ndcg_at_k(relevances, k=5))

        # Overall
        overall_ndcg = np.mean(ndcgs)

        # By category
        category_ndcgs = {}
        for cat in merged['category'].unique():
            cat_data = merged[merged['category'] == cat]
            cat_ndcgs = []
            for qid in cat_data['query_id'].unique():
                query_data = cat_data[cat_data['query_id'] == qid].sort_values('rank')
                relevances = query_data['relevance'].tolist()
                cat_ndcgs.append(ndcg_at_k(relevances, k=5))
            category_ndcgs[cat] = np.mean(cat_ndcgs) if cat_ndcgs else 0.0

        results.append({
            'configuration': config,
            'overall_ndcg': overall_ndcg,
            'num_queries': len(ndcgs),
            **{f'ndcg_{cat}': val for cat, val in category_ndcgs.items()}
        })

    # Agentic configuration
    agentic_merged = pd.merge(
        agentic_preds,
        existing_labels[['query_id', 'movieId', 'relevance', 'category']],
        on=['query_id', 'movieId'],
        how='inner'
    )

    if len(agentic_merged) > 0:
        # Overall
        ndcgs = []
        for qid in agentic_merged['query_id'].unique():
            query_data = agentic_merged[agentic_merged['query_id'] == qid].sort_values('rank')
            relevances = query_data['relevance'].tolist()
            ndcgs.append(ndcg_at_k(relevances, k=5))

        overall_ndcg = np.mean(ndcgs)

        # By category
        category_ndcgs = {}
        for cat in agentic_merged['category'].unique():
            cat_data = agentic_merged[agentic_merged['category'] == cat]
            cat_ndcgs = []
            for qid in cat_data['query_id'].unique():
                query_data = cat_data[cat_data['query_id'] == qid].sort_values('rank')
                relevances = query_data['relevance'].tolist()
                cat_ndcgs.append(ndcg_at_k(relevances, k=5))
            category_ndcgs[cat] = np.mean(cat_ndcgs) if cat_ndcgs else 0.0

        results.append({
            'configuration': 'agentic',
            'overall_ndcg': overall_ndcg,
            'num_queries': len(ndcgs),
            **{f'ndcg_{cat}': val for cat, val in category_ndcgs.items()}
        })

    # Print results
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("OVERALL NDCG@5 (UNBIASED)")
    print("=" * 70)
    print(results_df[['configuration', 'overall_ndcg', 'num_queries']].to_string(index=False))

    # Category comparison
    print("\n" + "=" * 70)
    print("CATEGORY-LEVEL NDCG@5")
    print("=" * 70)

    # Get all category columns
    cat_cols = [col for col in results_df.columns if col.startswith('ndcg_')]

    if cat_cols:
        print("\nConfiguration | " + " | ".join([col.replace('ndcg_', '') for col in cat_cols]))
        print("-" * 70)
        for _, row in results_df.iterrows():
            config = row['configuration']
            values = " | ".join([f"{row[col]:.4f}" if pd.notna(row[col]) else "N/A" for col in cat_cols])
            print(f"{config:15} | {values}")

    # Focus on Agentic vs Best Classic
    print("\n" + "=" * 70)
    print("AGENTIC vs CLASSIC (HYBRID_WITH_CE) COMPARISON")
    print("=" * 70)

    if 'agentic' in results_df['configuration'].values:
        agentic_row = results_df[results_df['configuration'] == 'agentic'].iloc[0]

        # Find hybrid_with_ce
        classic_row = results_df[results_df['configuration'] == 'hybrid_with_ce']

        if len(classic_row) > 0:
            classic_row = classic_row.iloc[0]

            print(f"\nOverall NDCG:")
            print(f"  Classic (hybrid_with_ce): {classic_row['overall_ndcg']:.4f}")
            print(f"  Agentic:                  {agentic_row['overall_ndcg']:.4f}")
            print(f"  Difference:               {agentic_row['overall_ndcg'] - classic_row['overall_ndcg']:+.4f}")

            print(f"\nBy Category:")
            for col in cat_cols:
                cat_name = col.replace('ndcg_', '')
                if pd.notna(agentic_row[col]) and pd.notna(classic_row[col]):
                    diff = agentic_row[col] - classic_row[col]
                    print(f"  {cat_name:20} | Classic: {classic_row[col]:.4f} | Agentic: {agentic_row[col]:.4f} | Diff: {diff:+.4f}")

    # Save results
    output_path = EVALUATION_DIR / 'ndcg_unbiased.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")

    # Warning message
    print("\n" + "=" * 70)
    print("⚠️  NOTE: These are UNBIASED results using only existing labels.")
    print("Previous results with LLM-generated labels were inflated by ~1.3 points.")
    print("=" * 70)

if __name__ == "__main__":
    compute_ndcg_unbiased()
