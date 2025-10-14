from __future__ import annotations

"""Hugging Face Hub client with rate limiting."""

import logging
from typing import TYPE_CHECKING, Any, Optional, cast
from urllib.parse import urlparse

import requests
from huggingface_hub import hf_hub_url

if TYPE_CHECKING:
    from huggingface_hub import HfApi, ModelInfo  # type: ignore[import]
    from huggingface_hub.errors import HfHubHTTPError  # type: ignore[import]
else:
    HfApi = Any  # type: ignore[assignment]
    ModelInfo = Any  # type: ignore[assignment]
    HfHubHTTPError = Exception  # type: ignore[assignment]

from src.clients.base_client import BaseClient
from src.net.rate_limiter import RateLimiter

DEFAULT_MAX_CALLS = 5
DEFAULT_PERIOD_SECONDS = 1.0


class HFClient(BaseClient[ModelInfo]):
    """Thin wrapper around ``huggingface_hub.HfApi`` with rate limiting."""

    def __init__(
        self,
        *,
        api: Optional[Any] = None,
        rate_limiter: Optional[RateLimiter] = None,
        logger: Optional[logging.Logger] = None,
        http_session: Optional[requests.Session] = None,
    ) -> None:
        limiter = rate_limiter or RateLimiter(
            max_calls=DEFAULT_MAX_CALLS,
            period_seconds=DEFAULT_PERIOD_SECONDS,
        )
        super().__init__(limiter, logger=logger)
        self._http_session = http_session or requests.Session()
        if api is not None:
            self._api = api
        else:
            try:
                from huggingface_hub import \
                    HfApi as RuntimeHfApi  # type: ignore[import]
            except ModuleNotFoundError as error:  # pragma: no cover
                message = (
                    "huggingface_hub is not installed. "
                    "Run './run install' first."
                )
                raise RuntimeError(message) from error

            self._api = cast(HfApi, RuntimeHfApi())

    def get_model_info(self, repo_id: str) -> ModelInfo:
        """Fetch model metadata from the Hugging Face Hub."""

        normalized_repo = self._normalize_repo_id(repo_id)

        def _operation() -> ModelInfo:
            return self._api.model_info(normalized_repo)

        return self._execute_with_rate_limit(
            _operation,
            name=f"hf.model_info({normalized_repo})",
        )

    def model_exists(self, repo_id: str) -> bool:
        """Return ``True`` when the model exists on the Hub."""
        try:
            self.get_model_info(repo_id)
        except HfHubHTTPError:
            return False
        return True

    def get_model_readme(self, repo_id: str) -> str:
        """Fetch the model card README as UTF-8 text."""

        normalized_repo = self._normalize_repo_id(repo_id)

        def _operation() -> str:
            url = hf_hub_url(
                repo_id=normalized_repo,
                filename="README.md",
            )
            response = self._http_session.get(url, timeout=30)
            if response.status_code == 404:
                return ""
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text

        try:
            return self._execute_with_rate_limit(
                _operation,
                name=f"hf.model_readme({normalized_repo})",
            )
        except (requests.RequestException, HfHubHTTPError):
            return ""

    @staticmethod
    def _normalize_repo_id(repo_identifier: str) -> str:
        trimmed = repo_identifier.strip()
        if not trimmed:
            raise ValueError("Repository identifier cannot be empty.")

        if "://" not in trimmed:
            return trimmed

        parsed = urlparse(trimmed)
        if parsed.netloc != "huggingface.co":
            raise ValueError(
                f"Unsupported Hugging Face host: {parsed.netloc}"
            )

        segments = [segment for segment in parsed.path.split("/") if segment]
        if not segments:
            raise ValueError(f"Unable to extract repo id from URL: {trimmed}")

        if segments[0] in {"datasets", "spaces", "models"}:
            segments = segments[1:]

        if len(segments) < 2:
            raise ValueError(f"Unable to extract repo id from URL: {trimmed}")

        return "/".join(segments[:2])
