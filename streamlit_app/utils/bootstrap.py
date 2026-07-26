"""Streamlit import bootstrap.

Ensures the project root is on sys.path so sibling packages like `data`
can be imported when Streamlit runs pages from `streamlit_app/`.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)

    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    return project_root
