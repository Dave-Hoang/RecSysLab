"""
Analyze diversity and novelty: How different are Agentic recommendations from Classic?

Measures:
1. Overlap: % movies recommended by both systems
2. Novelty: % movies unique to Agentic
3. Category distribution: Where does Agentic explore more?
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import EVALUATION_DIR

def analyze_diversity():
    # Load predictions
    classic = pd.read_csv(EVALUATION_DIR / 'predictions.csv')
    agentic = pd.read_csv(EVALUATION_DIR / 'agentic_predictions.csv')
    labels = pd.read_csv(EVALUATION_DIR / 'labels.csv')

    # Focus on best Classic config
    classic_best = classic[classic['configuration'] == 'hybrid_with_ce'].copy()

    print("=" * 70)
    print("DIVERSITY & NOVELTY ANALYSIS")
    print("=" * 70)

    # Overall statistics
    print("\n--- Overall Statistics ---")
    print(f"Classic (hybrid_with_ce) predictions: {len(classic_best)}")
    print(f"Agentic predictions: {len(agentic)}")

    # Unique movies
    classic_movies = set(classic_best['movieId'].unique())
    agentic_movies = set(agentic['movieId'].unique())

    overlap_movies = classic_movies & agentic_movies
    classic_only = classic_movies - agentic_movies
    agentic_only = agentic_movies - classic_movies

    print(f"\nUnique movies:")
    print(f"  Classic: {len(classic_movies)}")
    print(f"  Agentic: {len(agentic_movies)}")
    print(f"  Overlap: {len(overlap_movies)} ({len(overlap_movies)/len(agentic_movies)*100:.1f}% of Agentic)")
    print(f"  Agentic-only (novel): {len(agentic_only)} ({len(agentic_only)/len(agentic_movies)*100:.1f}% of Agentic)")

    # Per-query overlap
    print("\n--- Per-Query Overlap Analysis ---")

    query_overlaps = []
    for qid in agentic['query_id'].unique():
        classic_q = set(classic_best[classic_best['query_id'] == qid]['movieId'])
        agentic_q = set(agentic[agentic['query_id'] == qid]['movieId'])

        overlap = len(classic_q & agentic_q)
        agentic_novel = len(agentic_q - classic_q)

        query_overlaps.append({
            'query_id': qid,
            'overlap': overlap,
            'agentic_novel': agentic_novel,
            'overlap_pct': overlap / len(agentic_q) * 100 if len(agentic_q) > 0 else 0
        })

    overlap_df = pd.DataFrame(query_overlaps)
    print(f"Average overlap per query: {overlap_df['overlap'].mean():.2f} / 5 movies")
    print(f"Average novel movies per query: {overlap_df['agentic_novel'].mean():.2f} / 5 movies")
    print(f"Queries with 100% overlap: {len(overlap_df[overlap_df['overlap'] == 5])}")
    print(f"Queries with 0% overlap: {len(overlap_df[overlap_df['overlap'] == 0])}")

    # Category-level analysis
    print("\n--- Category-Level Novelty ---")

    for cat in agentic['category'].unique():
        cat_movies = set(agentic[agentic['category'] == cat]['movieId'])
        cat_classic = set(classic_best[classic_best['query_id'].isin(
            agentic[agentic['category'] == cat]['query_id']
        )]['movieId'])

        cat_novel = cat_movies - cat_classic
        novelty_pct = len(cat_novel) / len(cat_movies) * 100 if len(cat_movies) > 0 else 0

        print(f"{cat:20} | Total: {len(cat_movies):3} | Novel: {len(cat_novel):3} ({novelty_pct:.1f}%)")

    # Label coverage analysis
    print("\n" + "=" * 70)
    print("LABEL COVERAGE ANALYSIS")
    print("=" * 70)

    # Which Agentic recommendations have existing labels?
    agentic_with_labels = pd.merge(
        agentic[['query_id', 'movieId']],
        labels[['query_id', 'movieId']],
        on=['query_id', 'movieId'],
        how='inner'
    )

    agentic_without_labels = len(agentic) - len(agentic_with_labels)

    print(f"\nAgentic predictions with existing labels: {len(agentic_with_labels)} ({len(agentic_with_labels)/len(agentic)*100:.1f}%)")
    print(f"Agentic predictions WITHOUT labels: {agentic_without_labels} ({agentic_without_labels/len(agentic)*100:.1f}%)")

    # Of those without labels, how many are novel (not in Classic)?
    agentic_novel_ids = list(agentic_only)
    agentic_novel_without_labels = agentic[
        agentic['movieId'].isin(agentic_novel_ids)
    ]

    # Check which have labels
    novel_with_labels = pd.merge(
        agentic_novel_without_labels[['query_id', 'movieId']],
        labels[['query_id', 'movieId']],
        on=['query_id', 'movieId'],
        how='inner'
    )

    print(f"\nOf the {len(agentic_only)} unique Agentic-only movies:")
    print(f"  Predictions using them: {len(agentic_novel_without_labels)}")
    print(f"  With existing labels: {len(novel_with_labels)}")
    print(f"  WITHOUT labels (need LLM judge): {len(agentic_novel_without_labels) - len(novel_with_labels)}")

    # Conclusion
    print("\n" + "=" * 70)
    print("EVALUATION BIAS ASSESSMENT")
    print("=" * 70)

    if len(agentic_only) > 20:
        print(f"""
🚨 SIGNIFICANT EVALUATION BIAS DETECTED

Agentic Mode explores {len(agentic_only)} movies that Classic never recommended.
These movies represent {len(agentic_only)/len(agentic_movies)*100:.1f}% of Agentic's unique movie pool.

Current evaluation (using Classic-generated ground truth) is BIASED because:
1. Ground truth was created from Classic's candidate pool
2. Agentic's novel discoveries are not fairly evaluated
3. {agentic_without_labels} Agentic predictions ({agentic_without_labels/len(agentic)*100:.1f}%) lack labels

RECOMMENDATION:
→ Use Pairwise Comparison (Option 2) instead of NDCG on biased ground truth
→ OR: Manually label the {agentic_without_labels} missing predictions
        """)
    else:
        print(f"""
✅ LOW EVALUATION BIAS

Agentic Mode mostly recommends movies that Classic also explored ({len(overlap_movies)} overlap).
Only {len(agentic_only)} novel movies ({len(agentic_only)/len(agentic_movies)*100:.1f}%).

Current ground truth is ACCEPTABLE for comparison, but still recommend:
→ Manually review the {agentic_without_labels} unlabeled predictions
        """)

if __name__ == "__main__":
    analyze_diversity()
