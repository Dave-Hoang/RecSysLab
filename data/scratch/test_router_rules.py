"""Quick test for Router Rule 1 & 2 (Negative Constraints & Emotion)."""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.graph.graph import build_graph
from src.retrieval.embeddings import load_embedding_model
from src.retrieval.vector_store import load_vector_store
from src.ranking.cross_encoder import load_cross_encoder

def test_router_classification():
    print("=" * 70)
    print("ROUTER RULE VALIDATION TEST")
    print("=" * 70)

    # Load dependencies
    print("\n[*] Loading models...")
    emb = load_embedding_model()
    vs = load_vector_store(embedding_model=emb)
    ce = load_cross_encoder()
    graph = build_graph(vector_store=vs, cross_encoder=ce)

    # Test cases with expected intent
    test_cases = [
        # Rule 1: Negative Constraints (expect: need_expansion)
        ("action movies without violence", "need_expansion"),
        ("not scary horror movies", "need_expansion"),
        ("thriller with no jump scares", "need_expansion"),

        # Rule 2: Emotion/Theme (expect: need_expansion)
        ("something to cry to", "need_expansion"),
        ("feel-good movies", "need_expansion"),
        ("dark and depressing films", "need_expansion"),

        # Rule 3: Multi-condition without negation (expect: search_movie)
        ("psychological sci-fi movies", "search_movie"),
        ("romantic comedy thriller", "search_movie"),
        ("action drama with mystery", "search_movie"),

        # Edge case: Multi + Negative (Rule 1 priority, expect: need_expansion)
        ("psychological sci-fi without horror", "need_expansion"),
        ("romantic comedy but not cheesy", "need_expansion"),
    ]

    results = []
    for query, expected_intent in test_cases:
        result = graph.invoke({
            "original_query": query,
            "top_n": 3,
            "include_explanation": False,
        })

        actual_intent = result.get("intent")
        status = "✅ PASS" if actual_intent == expected_intent else "❌ FAIL"

        results.append({
            "query": query,
            "expected": expected_intent,
            "actual": actual_intent,
            "status": status,
            "reasoning": result.get("intent_reasoning", ""),
            "path": " → ".join(result.get("execution_path", [])),
        })

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    pass_count = 0
    fail_count = 0

    for r in results:
        print(f"\n{r['status']} Query: {r['query']}")
        print(f"   Expected: {r['expected']} | Actual: {r['actual']}")
        print(f"   Reasoning: {r['reasoning']}")
        print(f"   Path: {r['path']}")

        if r['status'] == "✅ PASS":
            pass_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: {pass_count}/{len(test_cases)} tests passed")
    print("=" * 70)

    if fail_count > 0:
        print(f"\n⚠️  {fail_count} test(s) failed. Router may need prompt tuning.")
    else:
        print("\n🎉 All tests passed! Router rules working as expected.")

if __name__ == "__main__":
    test_router_classification()
