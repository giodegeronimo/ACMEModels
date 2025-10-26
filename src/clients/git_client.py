from __future__ import annotations

"""Git service client with rate limiting support."""

import logging
from typing import Any, Optional, Protocol, cast

import requests  # type: ignore[import]

from src.clients.base_client import BaseClient
from src.net.rate_limiter import RateLimiter

DEFAULT_MAX_CALLS = 5
DEFAULT_PERIOD_SECONDS = 1.0


class _SessionWithGet(Protocol):
    def get(self, url: str, timeout: int) -> Any: ...


class GitClient(BaseClient[Any]):
    """Generic adapter for interacting with Git repository hosts."""

    def __init__(
        self,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        logger: Optional[logging.Logger] = None,
        session: Optional[_SessionWithGet] = None,
    ) -> None:
        limiter = rate_limiter or RateLimiter(
            max_calls=DEFAULT_MAX_CALLS,
            period_seconds=DEFAULT_PERIOD_SECONDS,
        )
        super().__init__(limiter, logger=logger)

        self._session: _SessionWithGet = cast(
            _SessionWithGet, session or requests.Session()
        )

    def get_repo_metadata(self, repo_url: str) -> dict[str, Any]:
        """Fetch repository metadata from supported hosts."""
        normalized = repo_url.strip().rstrip("/")
        if normalized.startswith("https://github.com/"):
            return self._execute_with_rate_limit(
                lambda: self._fetch_github_repo(normalized),
                name=f"github.repo({normalized})",
            )

        raise ValueError(f"Unsupported git repository host: {repo_url}")

    def list_repo_files(
        self,
        repo_url: str,
        *,
        branch: Optional[str] = None,
    ) -> list[str]:
        """Return the file paths for the repository's tree.

        Currently supports GitHub repositories using the git trees API.
        """

        normalized = repo_url.strip().rstrip("/")
        if normalized.startswith("https://github.com/"):
            return self._execute_with_rate_limit(
                lambda: self._fetch_github_tree(normalized, branch),
                name=f"github.tree({normalized})",
            )

        raise ValueError(f"Unsupported git repository host: {repo_url}")

    def list_repo_contributors(
        self,
        repo_url: str,
        *,
        per_page: int = 100,
    ) -> list[dict[str, Any]]:
        """Return a list of contributors for the repository.

        Each contributor dictionary contains at least ``login`` and
        ``contributions`` keys. Currently supports GitHub repositories.
        """

        normalized = repo_url.strip().rstrip("/")
        if normalized.startswith("https://github.com/"):
            return self._execute_with_rate_limit(
                lambda: self._fetch_github_contributors(normalized, per_page),
                name=f"github.contributors({normalized})",
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

    def _fetch_github_tree(
        self,
        repo_url: str,
        branch: Optional[str],
    ) -> list[str]:
        parts = repo_url.removeprefix("https://github.com/").split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid GitHub repository URL: {repo_url}")

        owner, repo = parts[0], parts[1]

        tree_branch = branch
        if tree_branch is None:
            metadata = self._fetch_github_repo(repo_url)
            tree_branch = metadata.get("default_branch") or "main"

        api_url = (
            "https://api.github.com/repos/"
            f"{owner}/{repo}/git/trees/{tree_branch}?recursive=1"
        )

        response = self._session.get(api_url, timeout=10)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to retrieve repo tree: {response.status_code}"
            )

        body = response.json()
        tree = body.get("tree", [])
        paths: list[str] = []
        for entry in tree:
            if isinstance(entry, dict) and entry.get("type") == "blob":
                path = entry.get("path")
                if isinstance(path, str):
                    paths.append(path)
        return paths

    def _fetch_github_contributors(
        self,
        repo_url: str,
        per_page: int,
    ) -> list[dict[str, Any]]:
        parts = repo_url.removeprefix("https://github.com/").split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid GitHub repository URL: {repo_url}")

        owner, repo = parts[0], parts[1]
        per_page = max(1, min(per_page, 100))
        api_url = (
            "https://api.github.com/repos/"
            f"{owner}/{repo}/contributors?per_page={per_page}&anon=1"
        )

        response = self._session.get(api_url, timeout=10)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to retrieve repo contributors: {response.status_code}"
            )

        payload = response.json()
        if not isinstance(payload, list):
            return []
        normalized: list[dict[str, Any]] = []
        for entry in payload:
            if isinstance(entry, dict):
                login = entry.get("login")
                contributions = entry.get("contributions")
                normalized.append(
                    {
                        "login": login,
                        "contributions": contributions,
                    }
                )
        return normalized
