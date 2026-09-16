import pandas as pd
import json
import hashlib
from datetime import datetime
import os
from pathlib import Path
from src.evaluation.metrics import ndcg_at_k

def compute_metrics():
    # 1. Validation
    queue = pd.read_csv('data/evaluation/annotation_queue_v1.csv')
    
    if len(queue) != 30:
        raise ValueError(f"Queue must have exactly 30 records, found {len(queue)}")
        
    if not set(queue['human_label'].dropna().unique()).issubset({0, 1, 2}):
        raise ValueError("human_label must be in {0, 1, 2}")
        
    if queue['human_label'].isna().any():
        raise ValueError("Missing human_label in queue")
        
    if not (queue['annotation_status'] == 'approved').all():
        raise ValueError("Not all annotation_status are approved")
        
    if queue.duplicated(subset=['query_id', 'movie_id']).any():
        raise ValueError("Duplicate query_id + movie_id in queue")

    print("✅ Validation successful")
    
    dist = queue['human_label'].value_counts(normalize=True) * 100
    counts = queue['human_label'].value_counts()
    print("\n--- Label Distribution in Delta Queue ---")
    for val in [0, 1, 2]:
        c = counts.get(val, 0)
        p = dist.get(val, 0)
        print(f"Relevance {val}: {c} ({p:.1f}%)")

    # 2. Merge Safe
    with open('data/evaluation/human_judgments_v1.jsonl', 'r', encoding='utf-8') as f:
        v1_records = [json.loads(line) for line in f]
        
    v1_keys = {(r['query_id'], r['movie_id']) for r in v1_records}
    
    queue_records = []
    for _, row in queue.iterrows():
        key = (str(row['query_id']), int(row['movie_id']))
        if key in v1_keys:
            raise ValueError(f"Duplicate key {key} found in v1")
            
        queue_records.append({
            "query_id": key[0],
            "movie_id": key[1],
            "relevance": int(row['human_label']),
            "notes": str(row['human_rationale']) if pd.notna(row['human_rationale']) else "",
            "metadata": {
                "label_source": "human",
                "annotator": "Duc Hoang",
                "rubric_version": "v1.1"
            }
        })
        
    v2_records = v1_records + queue_records
    
    if len(v2_records) != 375:
        raise ValueError(f"Expected 375 records, got {len(v2_records)}")
        
    v2_path = 'data/evaluation/human_judgments_v2.jsonl'
    with open(v2_path, 'w', encoding='utf-8') as f:
        for r in v2_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    with open(v2_path, 'rb') as f:
        v2_hash = hashlib.sha256(f.read()).hexdigest()
        
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "input_v1_count": len(v1_records),
        "input_queue_count": len(queue_records),
        "output_v2_count": len(v2_records),
        "output_v2_sha256": v2_hash,
        "annotator": "Duc Hoang",
        "rubric_versions": ["v1", "v1.1"]
    }
    with open('data/evaluation/merge_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        
    print("\n✅ Merge successful, generated v2 and manifest")

    # Save to labels_v2.csv for easy processing
    pd.DataFrame([{
        'query_id': r['query_id'], 
        'movieId': r['movie_id'], 
        'relevance': r['relevance'],
        'notes': r['notes']
    } for r in v2_records]).to_csv('data/evaluation/labels_v2.csv', index=False)

    # 3 & 4. Recompute Metrics
    queries = pd.read_csv('data/evaluation/queries.csv')
    labels = pd.read_csv('data/evaluation/labels_v2.csv')
    
    classic_preds = pd.read_csv('data/evaluation/predictions.csv')
    classic_preds = classic_preds[classic_preds['configuration'] == 'hybrid_with_ce']
    classic_preds['configuration'] = 'Classic'
    
    agentic_preds = pd.read_csv('data/evaluation/agentic_predictions.csv')
    agentic_preds['configuration'] = 'Agentic'
    
    all_preds = pd.concat([classic_preds, agentic_preds])
    
    # Merge with labels
    merged = all_preds.merge(labels[['query_id', 'movieId', 'relevance']], on=['query_id', 'movieId'], how='left')
    
    # Recompute Top-5 Metrics
    records = []
    pool = labels.groupby('query_id')['relevance'].apply(lambda x: sorted(x.tolist(), reverse=True)).to_dict()
    
    for (qid, conf), group in merged.groupby(['query_id', 'configuration']):
        group = group.sort_values('rank').head(5)
        relevances = group['relevance'].tolist()
        ideal = pool.get(qid, [])
        
        ndcg = ndcg_at_k(relevances, ideal, k=5, strict_mode=True)
        if any(pd.isna(x) for x in relevances):
            prec = float('nan')
        else:
            prec = sum(1 for x in relevances if x >= 1) / 5.0
            
        cat = queries[queries['query_id'] == qid]['category'].iloc[0]
        
        records.append({
            'query_id': qid,
            'category': cat,
            'configuration': conf,
            'ndcg_at_5': ndcg,
            'precision_at_5': prec
        })
        
    res = pd.DataFrame(records)
    
    print("\n--- Metric Recomputation Report ---")
    agg = res.groupby('configuration').agg(
        ndcg_at_5_mean=('ndcg_at_5', 'mean'),
        ndcg_at_5_median=('ndcg_at_5', 'median'),
        ndcg_at_5_std=('ndcg_at_5', 'std'),
        precision_at_5_mean=('precision_at_5', 'mean'),
        incomplete_queries=('ndcg_at_5', lambda x: x.isna().sum())
    )
    print("\nOverall Metrics:")
    print(agg.to_string())
    
    # Category breakdown
    print("\nBreakdown by Category (nDCG@5 Mean):")
    cat_agg = res.groupby(['category', 'configuration'])['ndcg_at_5'].mean().unstack()
    print(cat_agg.to_string())
    
    # Win/Loss
    classic = res[res['configuration'] == 'Classic'].set_index('query_id')['ndcg_at_5']
    agentic = res[res['configuration'] == 'Agentic'].set_index('query_id')['ndcg_at_5']
    
    diff = agentic - classic
    wins = (diff > 0).sum()
    ties = (diff == 0).sum()
    losses = (diff < 0).sum()
    
    print(f"\nPer-query Win/Tie/Loss (Agentic vs Classic, Human nDCG@5): {wins}/{ties}/{losses}")
    print("\nNote: Human ranking quality (nDCG/Precision) is evaluated here.")
    print("Metrics such as LLM-judged pairwise win rate, novelty discovery, and latency are evaluated separately.")
    
    coverage_c = merged[merged['configuration'] == 'Classic']['relevance'].notna().mean() * 100
    coverage_a = merged[merged['configuration'] == 'Agentic']['relevance'].notna().mean() * 100
    print(f"\nLabel Coverage after merge: Classic {coverage_c:.1f}%, Agentic {coverage_a:.1f}%")

if __name__ == '__main__':
    compute_metrics()
