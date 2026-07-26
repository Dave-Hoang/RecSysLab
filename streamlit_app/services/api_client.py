from typing import Any

import requests

from utils.config import (
    API_BASE_URL,
    DEFAULT_MODE,
    DEFAULT_TOP_K,
    REQUEST_TIMEOUT,
)


class APIClient:
    def __init__(self):
        self.base_url = API_BASE_URL

    def health(self) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/health",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def recommend(
        self,
        query: str,
        mode: str = DEFAULT_MODE,
        top_k: int = DEFAULT_TOP_K,
        include_explanation: bool = True,
    ) -> dict[str, Any]:

        payload = {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "include_explanation": include_explanation,
        }

        response = requests.post(
            f"{self.base_url}/recommendations",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return response.json()


api_client = APIClient()