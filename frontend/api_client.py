import os

import requests


class AssistantAPIError(RuntimeError):
    pass


class AssistantAPIClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

    def ask(self, question: str) -> dict:
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"question": question},
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise AssistantAPIError(
                "The cloud assistant is temporarily unavailable. Please try again in a moment."
            ) from exc

