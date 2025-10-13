from __future__ import annotations

"""Client for Purdue Gen AI Studio APIs with rate limiting."""

import logging
import os
from typing import Any, Dict, Optional

import requests  # type: ignore[import]

from src.clients.base_client import BaseClient
from src.net.rate_limiter import RateLimiter
from src.utils.env import load_dotenv

DEFAULT_MAX_CALLS = 3
DEFAULT_PERIOD_SECONDS = 1.0
DEFAULT_BASE_URL = "https://genai.rcac.purdue.edu/api"
CHAT_COMPLETIONS_PATH = "/chat/completions"
ENV_TOKEN_KEY = "GEN_AI_STUDIO_API_KEY"


class PurdueClient(BaseClient[Dict[str, Any]]):
    """Minimal wrapper around the Purdue GenAI Studio REST API."""

    def __init__(
        self,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        logger: Optional[logging.Logger] = None,
        session: Optional[requests.Session] = None,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        load_dotenv()

        limiter = rate_limiter or RateLimiter(
            max_calls=DEFAULT_MAX_CALLS,
            period_seconds=DEFAULT_PERIOD_SECONDS,
        )
        super().__init__(limiter, logger=logger)

        self._session = session or requests.Session()
        self._base_url = base_url.rstrip("/")
        self._api_token = os.environ.get(ENV_TOKEN_KEY)
        if not self._api_token:
            message = (
                f"{ENV_TOKEN_KEY} is not set. "
                "Populate it via the environment or a .env file."
            )
            raise RuntimeError(message)

    def generate_completion(
        self,
        prompt: str,
        *,
        model: str = "llama3.1:latest",
        stream: bool = False,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Request a text completion from the Purdue GenAI Studio service."""

        url = f"{self._base_url}{CHAT_COMPLETIONS_PATH}"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": stream,
            "temperature": 0.0,
        }
        payload.update(extra)

        def _operation() -> Dict[str, Any]:
            headers = self._build_headers(json_content=True)
            response = self._session.post(
                url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    "Purdue API returned "
                    f"{response.status_code}: {response.text}"
                )
            return response.json()

        return self._execute_with_rate_limit(
            _operation,
            name="purdue.generate_completion",
        )

    def llm(
        self,
        prompt: str,
        *,
        model: str = "llama3.1:latest",
        stream: bool = False,
        **extra: Any,
    ) -> str:
        """Return the assistant's text response for a prompt."""

        completion = self.generate_completion(
            prompt,
            model=model,
            stream=stream,
            **extra,
        )

        try:
            choices = completion["choices"]
            first_choice = choices[0]
            message = first_choice.get("message") or {}
            content = message.get("content")
            if content is None and "delta" in first_choice:
                content = first_choice["delta"].get("content")
        except (KeyError, IndexError, TypeError) as error:  # pragma: no cover
            raise RuntimeError(
                "Unexpected Purdue API response structure."
            ) from error

        if content is None:
            raise RuntimeError("Missing content in Purdue API response.")

        return content

    def _build_headers(self, json_content: bool = False) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Accept": "application/json",
        }
        if json_content:
            headers["Content-Type"] = "application/json"
        return headers
