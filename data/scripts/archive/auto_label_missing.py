import pandas as pd
import sys
import os
import json
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.graph.nodes import _get_gemini_llm, _extract_text

def label_missing():
    preds = pd.read_csv('d:/Project/RecSysLab/data/evaluation/agentic_predictions.csv')
    labels_path = Path('d:/Project/RecSysLab/data/evaluation/labels.csv')
    labels = pd.read_csv(labels_path)
    movies = pd.read_csv('d:/Project/RecSysLab/data/ml-32m/movies.csv')
    
    # Remove previously failed labels
    labels = labels[labels['notes'] != 'LLM parsing error']
    
    # Identify missing
    merged = pd.merge(preds, labels, on=['query_id', 'movieId'], how='left')
    missing = merged[merged['relevance'].isna()].copy()
    
    if missing.empty:
        print("No missing labels found.")
        # Still write back labels to be safe
        labels.to_csv(labels_path, index=False)
        return
        
    print(f"Found {len(missing)} missing query-movie pairs to label.")
    
    # Merge with movies to get genres
    missing = pd.merge(missing, movies[['movieId', 'genres']], on='movieId', how='left')
    
    missing['query'] = missing['query_x']
    missing['title'] = missing['title_x']
    missing['category'] = missing['category_x']
    missing['difficulty'] = missing['difficulty_x']
    if 'genres_y' in missing.columns:
        missing['genres'] = missing['genres_y']
        
    llm = _get_gemini_llm()
    new_labels = []
    
    prompt_template = """
    You are an expert movie recommendation judge. 
    Evaluate the relevance of a movie to a user's query.
    
    User Query: "{query}"
    Movie Title: "{title}"
    Movie Genres: "{genres}"
    
    Rules:
    - Relevance 2 (Highly Relevant): The movie perfectly matches the user's intent, constraints, and themes. If the user explicitly asks to exclude something (e.g. "without ghosts", "not scary"), and the movie likely has it, relevance cannot be 2.
    - Relevance 1 (Partially Relevant): The movie matches some aspects of the query, but misses a key constraint or is only somewhat related.
    - Relevance 0 (Irrelevant): The movie directly violates the user's explicit constraints (e.g., contains ghosts when asked "without ghosts", or is a comedy when asked "no comedy"), or is completely unrelated.
    
    Output ONLY a valid JSON object in this format:
    {{"relevance": 2, "reason": "..."}}
    """
    
    for idx, row in tqdm(missing.iterrows(), total=len(missing)):
        prompt = prompt_template.format(
            query=row['query'], 
            title=row['title'], 
            genres=row['genres']
        )
        
        try:
            response = llm.invoke(prompt)
            content = _extract_text(response.content)
            
            if content.startswith('```json'):
                content = '\n'.join(content.splitlines()[1:-1])
            elif content.startswith('```'):
                content = '\n'.join(content.splitlines()[1:-1])
            
            parsed = json.loads(content)
            relevance = int(parsed.get('relevance', 0))
            reason = parsed.get('reason', '')
        except Exception as e:
            print(f"Failed to parse LLM output for {row['title']}: {e}")
            relevance = 0
            reason = "LLM parsing error"
            
        new_labels.append({
            'query_id': row['query_id'],
            'query': row['query'],
            'category': row['category'],
            'difficulty': row['difficulty'],
            'movieId': row['movieId'],
            'title': row['title'],
            'genres': row['genres'],
            'relevance': relevance,
            'notes': reason
        })
        
    new_labels_df = pd.DataFrame(new_labels)
    updated_labels = pd.concat([labels, new_labels_df], ignore_index=True)
    
    updated_labels = updated_labels.drop_duplicates(subset=['query_id', 'movieId'], keep='last')
    updated_labels.to_csv(labels_path, index=False)
    
    print(f"Added {len(new_labels_df)} new labels to {labels_path}.")

if __name__ == "__main__":
    label_missing()
