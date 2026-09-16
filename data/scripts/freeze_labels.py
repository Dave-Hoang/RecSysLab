import pandas as pd
import json
from pathlib import Path

def freeze_labels():
    labels_csv = Path('data/evaluation/labels.csv')
    output_jsonl = Path('data/evaluation/human_judgments_v1.jsonl')
    
    if not labels_csv.exists():
        print(f"File not found: {labels_csv}")
        return
        
    df = pd.read_csv(labels_csv)
    
    records = []
    for _, row in df.iterrows():
        record = {
            "query_id": str(row['query_id']),
            "movie_id": int(row['movieId']),
            "relevance": int(row['relevance']),
            "notes": str(row['notes']) if pd.notna(row['notes']) else "",
            "metadata": {
                "label_source": "human",
                "annotator": "Duc Hoang",
                "rubric_version": "v1"
            }
        }
        if 'category' in row:
            record['category'] = str(row['category'])
        records.append(record)
        
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    print(f"Successfully froze {len(records)} labels to {output_jsonl}")

if __name__ == '__main__':
    freeze_labels()
