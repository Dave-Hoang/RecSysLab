import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add data to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ranking.cross_encoder import get_cross_encoder

# Define diverse query-document pairs with manual labels
pairs_data = [
    # Relevant pairs
    ("space movie", "Title: Interstellar. Genres: Sci-Fi, Drama", 1),
    ("space movie", "Title: 2001: A Space Odyssey. Genres: Sci-Fi", 1),
    ("space movie", "Title: Gravity. Genres: Sci-Fi, Thriller", 1),
    ("romantic comedy", "Title: When Harry Met Sally. Genres: Romance, Comedy", 1),
    ("romantic comedy", "Title: Crazy Rich Asians. Genres: Romance, Comedy", 1),
    ("superhero action", "Title: The Avengers. Genres: Action, Sci-Fi", 1),
    ("superhero action", "Title: Spider-Man. Genres: Action, Adventure", 1),
    ("horror ghosts", "Title: The Conjuring. Genres: Horror, Thriller", 1),
    ("horror ghosts", "Title: Insidious. Genres: Horror, Mystery", 1),
    ("animated family", "Title: Toy Story. Genres: Animation, Family", 1),
    ("animated family", "Title: Finding Nemo. Genres: Animation, Family", 1),
    ("world war 2", "Title: Saving Private Ryan. Genres: Action, Drama, War", 1),
    ("world war 2", "Title: Schindler's List. Genres: Biography, Drama, History", 1),
    ("heist movie", "Title: Ocean's Eleven. Genres: Crime, Thriller", 1),
    ("heist movie", "Title: Inception. Genres: Action, Sci-Fi, Thriller", 1), # loosely heist
    
    # Irrelevant pairs
    ("space movie", "Title: Toy Story. Genres: Animation, Family", 0),
    ("space movie", "Title: The Conjuring. Genres: Horror, Thriller", 0),
    ("space movie", "Title: Titanic. Genres: Romance, Drama", 0),
    ("romantic comedy", "Title: Saw. Genres: Horror, Thriller", 0),
    ("romantic comedy", "Title: The Terminator. Genres: Action, Sci-Fi", 0),
    ("romantic comedy", "Title: Schindler's List. Genres: Biography, Drama, History", 0),
    ("superhero action", "Title: The Notebook. Genres: Romance, Drama", 0),
    ("superhero action", "Title: Finding Nemo. Genres: Animation, Family", 0),
    ("horror ghosts", "Title: Crazy Rich Asians. Genres: Romance, Comedy", 0),
    ("horror ghosts", "Title: The Avengers. Genres: Action, Sci-Fi", 0),
    ("animated family", "Title: Saw. Genres: Horror, Thriller", 0),
    ("animated family", "Title: Saving Private Ryan. Genres: Action, Drama, War", 0),
    ("world war 2", "Title: Toy Story. Genres: Animation, Family", 0),
    ("world war 2", "Title: The Hangover. Genres: Comedy", 0),
    ("heist movie", "Title: The Notebook. Genres: Romance, Drama", 0),
    ("heist movie", "Title: Finding Nemo. Genres: Animation, Family", 0),
]

def main():
    model = get_cross_encoder()
    
    queries = [p[0] for p in pairs_data]
    docs = [p[1] for p in pairs_data]
    labels = [p[2] for p in pairs_data]
    
    pairs_for_model = [[q, d] for q, d in zip(queries, docs)]
    
    print(f"Running predictions on {len(pairs_for_model)} pairs...")
    # Bypass sigmoid to get raw logits
    raw_logits = model.predict(pairs_for_model, show_progress_bar=False, activation_fct=lambda x: x)
    
    df = pd.DataFrame({
        "query": queries,
        "document": docs,
        "raw_logit": raw_logits,
        "label": labels
    })
    
    out_path = Path(__file__).resolve().parent / "ce_logit_distribution.csv"
    df.to_csv(out_path, index=False)
    
    rel_logits = df[df["label"] == 1]["raw_logit"]
    irrel_logits = df[df["label"] == 0]["raw_logit"]
    
    print("\n--- STATISTICS ---")
    print("Relevant Group (n={}):".format(len(rel_logits)))
    print("  Mean: {:.4f}".format(rel_logits.mean()))
    print("  Std:  {:.4f}".format(rel_logits.std()))
    print("  Min:  {:.4f}".format(rel_logits.min()))
    print("  Max:  {:.4f}".format(rel_logits.max()))
    print("  p25:  {:.4f}".format(np.percentile(rel_logits, 25)))
    print("  p50:  {:.4f}".format(np.percentile(rel_logits, 50)))
    print("  p75:  {:.4f}".format(np.percentile(rel_logits, 75)))
    
    print("\nIrrelevant Group (n={}):".format(len(irrel_logits)))
    print("  Mean: {:.4f}".format(irrel_logits.mean()))
    print("  Std:  {:.4f}".format(irrel_logits.std()))
    print("  Min:  {:.4f}".format(irrel_logits.min()))
    print("  Max:  {:.4f}".format(irrel_logits.max()))
    print("  p25:  {:.4f}".format(np.percentile(irrel_logits, 25)))
    print("  p50:  {:.4f}".format(np.percentile(irrel_logits, 50)))
    print("  p75:  {:.4f}".format(np.percentile(irrel_logits, 75)))
    
    print(f"\nResults saved to {out_path}")
    
    proposed_floor = np.percentile(irrel_logits, 75)
    print(f"\n[PROPOSAL] Recommended CROSS_ENCODER_RAW_LOGIT_FLOOR (based on Irrelevant p75): {proposed_floor:.4f}")

if __name__ == "__main__":
    main()
