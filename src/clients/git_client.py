from __future__ import annotations

"""Git service client with rate limiting support."""

import logging
from typing import Any, Optional

import requests  # type: ignore[import]

from src.clients.base_client import BaseClient
from src.net.rate_limiter import RateLimiter

DEFAULT_MAX_CALLS = 5
DEFAULT_PERIOD_SECONDS = 1.0


class GitClient(BaseClient[Any]):
    """Generic adapter for interacting with Git repository hosts."""

    def __init__(
        self,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        logger: Optional[logging.Logger] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        limiter = rate_limiter or RateLimiter(
            max_calls=DEFAULT_MAX_CALLS,
            period_seconds=DEFAULT_PERIOD_SECONDS,
        )
        super().__init__(limiter, logger=logger)

        self._session = session or requests.Session()

    def get_repo_metadata(self, repo_url: str) -> dict[str, Any]:
        """Fetch repository metadata from supported hosts."""
        normalized = repo_url.strip().rstrip("/")
        if normalized.startswith("https://github.com/"):
            return self._execute_with_rate_limit(
                lambda: self._fetch_github_repo(normalized),
                name=f"github.repo({normalized})",
            )

        raise ValueError(f"Unsupported git repository host: {repo_url}")

    def _fetch_github_repo(self, repo_url: str) -> dict[str, Any]:
        parts = repo_url.removeprefix("https://github.com/").split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid GitHub repository URL: {repo_url}")

        owner, repo = parts[0], parts[1]
        api_url = f"https://api.github.com/repos/{owner}/{repo}"

        response = self._session.get(api_url, timeout=10)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to retrieve repo metadata: {response.status_code}"
            )

        return response.json()
