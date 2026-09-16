import pandas as pd

def generate_queue():
    queries = pd.read_csv('data/evaluation/queries.csv')
    classic_preds = pd.read_csv('data/evaluation/predictions.csv')
    classic_preds = classic_preds[classic_preds['configuration'] == 'hybrid_with_ce']
    agentic_preds = pd.read_csv('data/evaluation/agentic_predictions.csv')
    labels = pd.read_csv('data/evaluation/labels.csv')

    annotation_queue = []

    for idx, row in queries.iterrows():
        qid = row['query_id']
        qtext = row['query']
        qcat = row['category']
        
        c_movies = classic_preds[classic_preds['query_id'] == qid]
        a_movies = agentic_preds[agentic_preds['query_id'] == qid]
        
        union_ids = set(c_movies['movieId'].tolist() + a_movies['movieId'].tolist())
        labeled_ids = set(labels[labels['query_id'] == qid]['movieId'].tolist())
        
        unlabeled_ids = union_ids - labeled_ids
        
        for m in unlabeled_ids:
            c_row = c_movies[c_movies['movieId'] == m]
            a_row = a_movies[a_movies['movieId'] == m]
            
            c_rank = int(c_row['rank'].iloc[0]) if not c_row.empty else None
            a_rank = int(a_row['rank'].iloc[0]) if not a_row.empty else None
            
            info_row = a_row.iloc[0] if not a_row.empty else c_row.iloc[0]
            
            annotation_queue.append({
                'query_id': qid,
                'query': qtext,
                'category': qcat,
                'movie_id': m,
                'title': info_row['title'],
                'genres': info_row.get('genres', ''),
                'source_classic': not c_row.empty,
                'source_agentic': not a_row.empty,
                'classic_rank': c_rank,
                'agentic_rank': a_rank,
                'semantic_score_relative': info_row.get('semantic_similarity', None),
                'cosine_similarity_raw': info_row.get('cosine_similarity_raw', None),
                'cross_encoder_score': info_row.get('cross_encoder_score', None),
                'final_score': info_row.get('evaluation_score', info_row.get('final_score', None)),
                'annotation_status': 'pending_human_review',
                'human_label': '',
                'human_rationale': ''
            })
            
    df_queue = pd.DataFrame(annotation_queue)
    columns_order = [
        'query_id', 'query', 'category', 'movie_id', 'title', 'genres',
        'source_classic', 'source_agentic', 'classic_rank', 'agentic_rank',
        'semantic_score_relative', 'cosine_similarity_raw',
        'cross_encoder_score', 'final_score', 'annotation_status',
        'human_label', 'human_rationale'
    ]
    
    # Fill missing columns with None
    for col in columns_order:
        if col not in df_queue.columns:
            df_queue[col] = None
            
    df_queue = df_queue[columns_order]
    df_queue.to_csv('data/evaluation/annotation_queue_v1.csv', index=False)
    print(f"Queue generated with {len(df_queue)} records.")

if __name__ == '__main__':
    generate_queue()
