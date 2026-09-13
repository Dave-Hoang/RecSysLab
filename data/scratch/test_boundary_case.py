import sys
import pandas as pd
from pathlib import Path

# Add data to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ranking.hybrid_ranker import compute_cross_encoder_score
from src.ranking.cross_encoder import get_cross_encoder
import logging
logging.basicConfig(level=logging.WARNING)

def main():
    model = get_cross_encoder()
    
    query = "world war 2"
    
    # A batch where the max logit is < -7.7343
    # Schindler's List gets -8.8539
    # The Hangover gets even lower
    candidates = pd.DataFrame({
        "title": ["Schindler's List", "The Hangover"],
        "genres": ["Biography, Drama, History", "Comedy"],
        "page_content": [
            "Title: Schindler's List. Genres: Biography, Drama, History",
            "Title: The Hangover. Genres: Comedy"
        ]
    })
    
    print(f"Testing Query: '{query}'")
    
    result_df = compute_cross_encoder_score(
        candidates=candidates,
        query=query,
        cross_encoder=model
    )
    
    print("\n--- RESULTS ---")
    for idx, row in result_df.iterrows():
        print(f"Doc {idx}: {row['title']} ({row['genres']})")
        print(f"  -> Cross-Encoder Score (Normalized): {row['cross_encoder_score']}")

if __name__ == "__main__":
    main()
