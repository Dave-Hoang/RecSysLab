# Evaluation Summary

## 1. Evaluation Objective

The evaluation compares four retrieval and ranking configurations for
a natural-language movie recommendation system:

1. `faiss_only`
2. `hybrid_no_ce`
3. `cross_encoder_only`
4. `hybrid_with_ce`

The primary objective is to determine which configuration provides the
best ranking quality while maintaining a practical trade-off between
quality, latency, and system complexity.

---

## 2. Benchmark Design

The benchmark contains:

- 30 manually designed evaluation queries
- 4 system configurations
- Top 5 recommendations per configuration
- 600 ranked prediction rows
- Human relevance labels using three levels:
  - `0`: irrelevant
  - `1`: partially relevant
  - `2`: strongly relevant

The evaluation queries cover:

- Genre queries
- Emotion and theme queries
- Multi-condition queries
- Natural-language queries
- Negative-constraint queries
- Similar-movie queries

Query difficulty is divided into:

- Easy
- Medium
- Hard

---

## 3. Evaluation Metrics

The following metrics are used:

- Precision@5
- NDCG@5
- MRR@5
- MRR Strong@5
- Hit Rate@5
- Irrelevant@5
- Mean Relevance@5

NDCG@5 is treated as the primary metric because it considers both
graded relevance and ranking position.

For Precision@5, MRR@5, and Hit Rate@5:

- Relevance `1` and `2` are considered relevant.

For MRR Strong@5:

- Only relevance `2` is considered strongly relevant.

---

## 4. Overall Results
- Note: All metrics @5
             | Precision | NDCG   | MRR    | MRR Strong | Irrelevant | Mean Relevance |
Hybrid + CE  | 0.9133    | 0.8284 | 0.9500 | 0.8844     | 0.4333     | 1.6267 |
Hybrid - CE  | 0.8800    | 0.7742 | 0.9667 | 0.8611     | 0.6000     | 1.5133 |
CE only      | 0.8733    | 0.7684 | 0.9111 | 0.8511     | 0.6333     | 1.5200 |
FAISS only   | 0.8600    | 0.7329 | 0.9278 | 0.7944     | 0.7000     | 1.4733 |

The `hybrid_with_ce` configuration achieved the best overall:

- Precision@5
- NDCG@5
- MRR Strong@5
- Irrelevant@5
- Mean Relevance@5

Although `hybrid_no_ce` achieved a slightly higher MRR@5, ordinary
MRR only checks the position of the first result with relevance greater
than or equal to 1. NDCG@5 and MRR Strong@5 better represent the
quality of the complete ranked list.

Compared with `faiss_only`, `hybrid_with_ce`:

- Improved NDCG@5 from `0.7329` to `0.8284`
- Improved Precision@5 from `0.8600` to `0.9133`
- Improved MRR Strong@5 from `0.7944` to `0.8844`
- Reduced irrelevant recommendations from `0.7000` to `0.4333`

This corresponds to approximately:

- 13.0% relative improvement in NDCG@5
- 38.1% reduction in irrelevant Top-5 recommendations

---

## 5. Results by Query Difficulty

| Difficulty | FAISS only | Hybrid no CE | Cross-Encoder only | Hybrid with CE |
| Easy       | 0.8430     | 0.8596       | 0.8529             | **0.8850**     |
| Medium     | 0.7319     | 0.7717       | 0.8161             | **0.8514**     |
| Hard       | 0.6640     | 0.7226       | 0.6625             | **0.7674**     |

Values represent mean NDCG@5.

`hybrid_with_ce` achieved the highest NDCG@5 for all three difficulty
levels.

The improvement is more meaningful for medium and hard queries, where
simple vector retrieval is less capable of handling nuanced intent and
multiple constraints.

---

## 6. Results by Query Category

### Emotion and Theme

`hybrid_with_ce` achieved the highest NDCG@5:

- Hybrid with CE: `0.9033`
- Cross-Encoder only: `0.8696`
- Hybrid no CE: `0.8519`
- FAISS only: `0.7743`

This indicates that Cross-Encoder reranking is useful for queries that
depend on emotional meaning or abstract themes.

### Genre

`hybrid_with_ce` achieved an NDCG@5 of `0.9009`.

However, the other configurations also performed relatively well,
showing that simple genre queries can often be handled effectively by
embedding retrieval.

### Multi-condition

`hybrid_with_ce` achieved an NDCG@5 of `0.7829`, compared with `0.5998`
for FAISS-only retrieval.

This was one of the clearest improvements, indicating that reranking is
especially valuable when a query contains several simultaneous
conditions.

### Natural Language

`cross_encoder_only` achieved the highest category NDCG@5 of `0.7684`,
while `hybrid_with_ce` achieved `0.6644`.

This suggests that the Cross-Encoder understands some natural-language
queries well, but additional popularity or rule-based scores can
occasionally alter an already effective ranking.

### Negative Constraints

`hybrid_no_ce` achieved the highest NDCG@5 of `0.8204`.

`cross_encoder_only` achieved only `0.4488`, indicating that the current
Cross-Encoder is not consistently reliable for expressions such as:

- `not`
- `without`
- `no`

Rule-assisted ranking remains useful for explicit negative constraints.

### Similar Movie

`hybrid_with_ce` achieved the highest NDCG@5 of `0.8992`.

FAISS-only retrieval also performed well at `0.8414`, showing that
embedding similarity is already effective for this query type.

---

## 7. Representative Query Cases

### Case 1 — Multi-condition query

Query:

`psychological sci-fi movies`

Results:

- FAISS only NDCG@5: `0.3977`
- Hybrid with CE NDCG@5: `0.7739`

The Cross-Encoder helps evaluate the complete relationship between the
psychological and science-fiction aspects instead of relying primarily
on vector similarity.

### Case 2 — Emotion and relationship query

Query:

`sad movie about father and son`

Results:

- FAISS only NDCG@5: `0.6399`
- Hybrid with CE NDCG@5: `1.0000`

The reranker better captures both the emotional tone and the father-son
relationship.

### Case 3 — Negative constraint

Query:

`horror movie without ghosts`

Results:

- Hybrid no CE NDCG@5: `1.0000`
- Hybrid with CE NDCG@5: `0.6886`
- Cross-Encoder only NDCG@5: `0.2364`

The Cross-Encoder does not consistently handle the word `without`.
Explicit rule-based constraint handling is more reliable in this case.

### Case 4 — Simple genre query

Query:

`dark supernatural horror movies`

Results:

- FAISS only NDCG@5: `0.8688`
- Hybrid with CE NDCG@5: `0.6608`

The query is sufficiently direct for embedding retrieval. Additional
reranking changes an already effective ordering and reduces quality.

---

## 8. Production Configuration Decision

The recommended default production configuration is:

`hybrid_with_ce`

Reasons:

- Best overall NDCG@5
- Best overall Precision@5
- Best MRR Strong@5
- Lowest average number of irrelevant recommendations
- Best NDCG@5 across easy, medium, and hard queries

A second lightweight mode is retained:

`hybrid_no_ce`

This mode is appropriate when:

- Lower latency is preferred
- The query is simple
- Cross-Encoder inference is unavailable
- Explicit rules are needed for negative constraints

The system does not currently perform automatic per-query configuration
routing. Configuration selection is exposed as a production mode rather
than predicted dynamically.

---

## 9. Production Modes

### Quality Mode

Configuration:

`hybrid_with_ce`

Recommended for:

- Natural-language recommendation requests
- Emotion or theme queries
- Multi-condition queries
- Medium and hard queries

### Fast Mode

Configuration:

`hybrid_no_ce`

Recommended for:

- Lower-latency requests
- Simple genre queries
- Explicit constraint queries
- Environments with limited compute resources

---

## 10. Limitations

The benchmark contains only 30 queries.

It is sufficient for:

- Portfolio demonstration
- Ablation comparison
- Engineering decision support

It is not intended to establish a general research benchmark.

Some categories contain only four or six queries, so category-level
results should be interpreted as practical observations rather than
universal conclusions.

The current system also does not automatically classify an incoming
query and select the optimal ranking configuration.

---

## 11. Final Conclusion

On a manually labeled benchmark of 30 natural-language queries and 600
ranked predictions, the Hybrid with Cross-Encoder configuration achieved
the best overall performance, reaching `0.8284` NDCG@5 and `0.9133`
Precision@5.

Compared with FAISS-only retrieval, it improved NDCG@5 by approximately
13.0% and reduced irrelevant Top-5 recommendations by approximately
38.1%.

For production, Hybrid with Cross-Encoder is selected as the default
quality configuration, while Hybrid without Cross-Encoder is retained
as a lower-latency alternative.