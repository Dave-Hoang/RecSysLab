"""
Pairwise Evaluation: Classic vs Agentic using LLM Judge.

For each query, compare recommendation sets side-by-side.
LLM judge picks winner based on relevance to user intent.

This is FAIR evaluation that doesn't depend on Classic's ground truth.
"""
import pandas as pd
import json
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import EVALUATION_DIR
from src.graph.nodes import _get_gemini_llm, _extract_text

PAIRWISE_PROMPT = """You are an expert movie recommendation judge.

User Query: "{query}"

Evaluate two recommendation systems by comparing their Top-5 movie lists:

**System A:**
{system_a_movies}

**System B:**
{system_b_movies}

Your task:
1. Determine which system's recommendations BETTER match the user's query intent
2. Consider: genre relevance, constraint satisfaction, diversity, quality
3. If query has negative constraints (e.g., "without violence"), check violations

Output ONLY valid JSON:
{{
  "winner": "A" or "B" or "tie",
  "confidence": 0-100 (how confident you are),
  "reason": "Brief explanation (2-3 sentences max)"
}}

CRITICAL: Output ONLY the JSON, no other text."""

def format_movie_list(movies_df):
    """Format movie list for LLM judge."""
    lines = []
    for idx, row in movies_df.iterrows():
        title = row['title']
        genres = row.get('genres', 'N/A')
        lines.append(f"{idx+1}. {title} ({genres})")
    return "\n".join(lines)

def evaluate_pairwise():
    # Load data
    classic = pd.read_csv(EVALUATION_DIR / 'predictions.csv')
    agentic = pd.read_csv(EVALUATION_DIR / 'agentic_predictions.csv')
    queries = pd.read_csv(EVALUATION_DIR / 'queries.csv')
    movies = pd.read_csv('d:/Project/RecSysLab/data/ml-32m/movies.csv')

    # Use best Classic config
    classic_best = classic[classic['configuration'] == 'hybrid_with_ce'].copy()

    # Merge with movie metadata
    classic_best = pd.merge(classic_best, movies[['movieId', 'genres']], on='movieId', how='left')
    agentic = pd.merge(agentic, movies[['movieId', 'genres']], on='movieId', how='left')

    print("=" * 70)
    print("PAIRWISE EVALUATION: CLASSIC vs AGENTIC")
    print("=" * 70)

    llm = _get_gemini_llm()
    results = []

    for _, query_row in tqdm(queries.iterrows(), total=len(queries), desc="Evaluating queries"):
        qid = query_row['query_id']
        query_text = query_row['query']
        category = query_row.get('category', 'unknown')

        # Get Top-5 from each system
        classic_top5 = classic_best[classic_best['query_id'] == qid].sort_values('rank').head(5)
        agentic_top5 = agentic[agentic['query_id'] == qid].sort_values('rank').head(5)

        if len(classic_top5) == 0 or len(agentic_top5) == 0:
            continue

        # Format for LLM
        system_a_text = format_movie_list(classic_top5)
        system_b_text = format_movie_list(agentic_top5)

        # Randomly swap A/B to reduce position bias
        import random
        if random.random() > 0.5:
            system_a_text, system_b_text = system_b_text, system_a_text
            swapped = True
        else:
            swapped = False

        prompt = PAIRWISE_PROMPT.format(
            query=query_text,
            system_a_movies=system_a_text,
            system_b_movies=system_b_text
        )

        # LLM Judge
        try:
            response = llm.invoke(prompt)
            content = _extract_text(response.content)

            # Parse JSON
            if content.startswith('```json'):
                content = '\n'.join(content.splitlines()[1:-1])
            elif content.startswith('```'):
                content = '\n'.join(content.splitlines()[1:-1])

            parsed = json.loads(content)

            winner_raw = parsed.get('winner', 'tie').lower()
            confidence = int(parsed.get('confidence', 50))
            reason = parsed.get('reason', '')

            # Unswap winner
            if swapped:
                if winner_raw == 'a':
                    winner = 'agentic'
                elif winner_raw == 'b':
                    winner = 'classic'
                else:
                    winner = 'tie'
            else:
                if winner_raw == 'a':
                    winner = 'classic'
                elif winner_raw == 'b':
                    winner = 'agentic'
                else:
                    winner = 'tie'

        except Exception as e:
            print(f"[!] Failed to parse LLM output for {qid}: {e}")
            winner = 'tie'
            confidence = 0
            reason = f"Parse error: {e}"

        results.append({
            'query_id': qid,
            'query': query_text,
            'category': category,
            'winner': winner,
            'confidence': confidence,
            'reason': reason
        })

    # Save results
    results_df = pd.DataFrame(results)
    output_path = EVALUATION_DIR / 'pairwise_evaluation.csv'
    results_df.to_csv(output_path, index=False)

    # Analysis
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    win_counts = results_df['winner'].value_counts()
    total = len(results_df)

    classic_wins = win_counts.get('classic', 0)
    agentic_wins = win_counts.get('agentic', 0)
    ties = win_counts.get('tie', 0)

    print(f"\nTotal queries evaluated: {total}")
    print(f"Classic wins: {classic_wins} ({classic_wins/total*100:.1f}%)")
    print(f"Agentic wins: {agentic_wins} ({agentic_wins/total*100:.1f}%)")
    print(f"Ties: {ties} ({ties/total*100:.1f}%)")

    if agentic_wins > classic_wins:
        advantage = agentic_wins - classic_wins
        print(f"\n🎉 Agentic Mode WINS by +{advantage} queries!")
    elif classic_wins > agentic_wins:
        advantage = classic_wins - agentic_wins
        print(f"\n⚠️  Classic Mode WINS by +{advantage} queries")
    else:
        print(f"\n🤝 TIE - Both systems perform equally")

    # Average confidence
    avg_conf = results_df['confidence'].mean()
    print(f"\nAverage confidence: {avg_conf:.1f}%")

    # Category breakdown
    print("\n" + "=" * 70)
    print("CATEGORY-LEVEL BREAKDOWN")
    print("=" * 70)

    for cat in results_df['category'].unique():
        cat_df = results_df[results_df['category'] == cat]
        cat_wins = cat_df['winner'].value_counts()

        classic_cat = cat_wins.get('classic', 0)
        agentic_cat = cat_wins.get('agentic', 0)
        tie_cat = cat_wins.get('tie', 0)

        print(f"\n{cat}:")
        print(f"  Classic: {classic_cat} | Agentic: {agentic_cat} | Tie: {tie_cat}")

        if agentic_cat > classic_cat:
            print(f"  → Agentic DOMINATES (+{agentic_cat - classic_cat})")
        elif classic_cat > agentic_cat:
            print(f"  → Classic DOMINATES (+{classic_cat - agentic_cat})")

    # High-confidence wins
    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE WINS (≥80% confidence)")
    print("=" * 70)

    high_conf = results_df[results_df['confidence'] >= 80]
    high_conf_wins = high_conf['winner'].value_counts()

    print(f"\nHigh-confidence decisions: {len(high_conf)}/{total}")
    print(f"  Classic: {high_conf_wins.get('classic', 0)}")
    print(f"  Agentic: {high_conf_wins.get('agentic', 0)}")
    print(f"  Tie: {high_conf_wins.get('tie', 0)}")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    agentic_win_rate = agentic_wins / total * 100

    if agentic_win_rate >= 60:
        print(f"""
✅ STRONG EVIDENCE FOR AGENTIC MODE

Agentic wins {agentic_win_rate:.1f}% of queries with avg confidence {avg_conf:.1f}%.

Phase 1 improvements (Router + Original Query for CE) are WORKING.

NEXT STEPS:
1. ✅ Agentic Mode is production-ready for deployment
2. 📊 (Optional) Manual labeling to get precise NDCG numbers
3. 🚀 Monitor real-world performance with A/B test
        """)
    elif agentic_win_rate >= 45:
        print(f"""
⚖️  MIXED RESULTS

Agentic wins {agentic_win_rate:.1f}%, roughly on par with Classic.

Phase 1 improvements show potential but need refinement.

NEXT STEPS:
1. 📊 Manual labeling 30 predictions to get accurate NDCG
2. 🔧 Analyze losing queries → tune Router rules or Expansion prompt
3. 🧪 Consider Phase 2: Adaptive Threshold
        """)
    else:
        print(f"""
⚠️  CLASSIC MODE STILL BETTER

Agentic only wins {agentic_win_rate:.1f}% of queries.

Phase 1 improvements not yet effective.

NEXT STEPS:
1. 🔍 Analyze Agentic losses → identify failure patterns
2. 🔧 Tune Router prompt or Expansion strategy
3. 🧪 Implement Phase 2: Adaptive Threshold
4. ❓ Consider reverting to Classic for production
        """)

    print(f"\n✅ Results saved to: {output_path}")

if __name__ == "__main__":
    evaluate_pairwise()
