import os
import sys
from pathlib import Path
import pytest

# Ensure TEST_MODE is set
os.environ["TEST_MODE"] = "true"
sys.path.append(str(Path(__file__).parent.parent))

from src.graph.prompts import EXPANSION_SYSTEM_PROMPT, EXPANSION_USER_TEMPLATE
from src.graph.nodes import _get_expansion_llm

def test_expansion_bias():
    print("Testing Query Expansion bias...")
    llm = _get_expansion_llm()
    
    cases = [
        ("I want an emotional movie that will stay with me for days", "Regression Test"),
        ("atmospheric horror with eerie sound design", "Test B"),
        ("sci-fi with philosophical themes", "Test C"),
        ("light comedy for the weekend", "Test D"),
        ("a musical with memorable songs", "Test E")
    ]
    
    for q, name in cases:
        messages = [
            ("system", EXPANSION_SYSTEM_PROMPT),
            ("human", EXPANSION_USER_TEMPLATE.format(query=q)),
        ]
        response = llm.invoke(messages)
        output = response.content.lower()
        print(f"\n--- {name} ---")
        print(f"Original: {q}")
        print(f"Expanded: {response.content}")
        
        # specific regression checks for case A
        if name == "Regression Test":
            forbidden = ["musical", "music", "score", "soundtrack", "cinematography", "visuals", "biopic", "historical", "war", "sci-fi"]
            found = [word for word in forbidden if word in output]
            if found:
                print(f"FAILED: Found forbidden words {found} in Regression Test output!")
            else:
                print("PASSED: No forbidden words in Regression Test.")

if __name__ == "__main__":
    test_expansion_bias()
