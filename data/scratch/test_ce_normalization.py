import sys
import pandas as pd
from pathlib import Path

# Add data to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ranking.hybrid_ranker import compute_cross_encoder_score
from src.ranking.cross_encoder import get_cross_encoder

def main():
    model = get_cross_encoder()
    
    query = "I want to watch a space movie"
    
    candidates = pd.DataFrame({
        "title": ["Interstellar", "Toy Story"],
        "genres": ["Sci-Fi", "Animation"],
        "page_content": [
            "Title: Interstellar. Genres: Sci-Fi, Drama. Plot: A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
            "Title: Toy Story. Genres: Animation, Comedy, Family. Plot: A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room."
        ]
    })
    
    print(f"Testing Query: '{query}'")
    
    # Run the real model through the new compute_cross_encoder_score function
    result_df = compute_cross_encoder_score(
        candidates=candidates,
        query=query,
        cross_encoder=model
    )
    
    print("\n--- RESULTS ---")
    for idx, row in result_df.iterrows():
        print(f"Doc {idx}: {row['title']} ({row['genres']})")
        print(f"  -> Cross-Encoder Score (Normalized): {row['cross_encoder_score']}")
        
    print("\nResolution Check:")
    diff = abs(result_df.loc[0, "cross_encoder_score"] - result_df.loc[1, "cross_encoder_score"])
    print(f"Score difference: {diff:.6f}")
    if diff > 0.5:
        print("[OK] Good resolution between relevant and irrelevant.")
    else:
        print("[WARNING] Low resolution between relevant and irrelevant.")

if __name__ == "__main__":
    main()
