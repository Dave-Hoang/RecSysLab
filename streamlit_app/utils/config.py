"""
Streamlit configuration.
"""

import os

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)

DEFAULT_MODE = "quality"

DEFAULT_TOP_K = 5

REQUEST_TIMEOUT = 120