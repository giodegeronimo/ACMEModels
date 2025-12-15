"""

ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, Protocol, cast

import requests  # type: ignore[import]

from src.clients.base_client import BaseClient
from src.net.rate_limiter import RateLimiter

DEFAULT_MAX_CALLS = 5
DEFAULT_PERIOD_SECONDS = 1.0


class _SessionWithGet(Protocol):
    """
    _SessionWithGet: Class description.
    """

    def get(
        self,
        url: str,
        timeout: int,
        headers: Optional[dict[str, str]] = None,
    ) -> Any: ...

    def post(
        self,
        url: str,
        json: Any,
        timeout: int,
        headers: Optional[dict[str, str]] = None,
    ) -> Any: ...


class GitClient(BaseClient[Any]):
    """Generic adapter for interacting with Git repository hosts."""

    def __init__(
        self,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        logger: Optional[logging.Logger] = None,
        session: Optional[_SessionWithGet] = None,
    ) -> None:
        """
        __init__: Function description.
        :param rate_limiter:
        :param logger:
        :param session:
        :returns:
        """

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
            headers = self._auth_headers()
            return self._execute_with_rate_limit(
                lambda: self._fetch_github_repo(normalized, headers=headers),
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
            headers = self._auth_headers()
            return self._execute_with_rate_limit(
                lambda: self._fetch_github_tree(
                    normalized, branch, headers=headers
                ),
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
            headers = self._auth_headers()
            return self._execute_with_rate_limit(
                lambda: self._fetch_github_contributors(
                    normalized, per_page, headers=headers
                ),
                name=f"github.contributors({normalized})",
            )

        raise ValueError(f"Unsupported git repository host: {repo_url}")

    def _fetch_github_repo(
        self, repo_url: str, *, headers: Optional[dict[str, str]] = None
    ) -> dict[str, Any]:
        """
        _fetch_github_repo: Function description.
        :param repo_url:
        :param headers:
        :returns:
        """

        parts = repo_url.removeprefix("https://github.com/").split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid GitHub repository URL: {repo_url}")

        owner, repo = parts[0], parts[1]
        api_url = f"https://api.github.com/repos/{owner}/{repo}"

        response = self._session.get(api_url, timeout=10, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to retrieve repo metadata: {response.status_code}"
            )

        return response.json()

    def _fetch_github_tree(
        self,
        repo_url: str,
        branch: Optional[str],
        *,
        headers: Optional[dict[str, str]] = None,
    ) -> list[str]:
        """
        _fetch_github_tree: Function description.
        :param repo_url:
        :param branch:
        :param headers:
        :returns:
        """

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

        response = self._session.get(api_url, timeout=10, headers=headers)
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
        *,
        headers: Optional[dict[str, str]] = None,
    ) -> list[dict[str, Any]]:
        """
        _fetch_github_contributors: Function description.
        :param repo_url:
        :param per_page:
        :param headers:
        :returns:
        """

        parts = repo_url.removeprefix("https://github.com/").split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid GitHub repository URL: {repo_url}")

        owner, repo = parts[0], parts[1]
        per_page = max(1, min(per_page, 100))
        api_url = (
            "https://api.github.com/repos/"
            f"{owner}/{repo}/contributors?per_page={per_page}&anon=1"
        )

        response = self._session.get(api_url, timeout=10, headers=headers)
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

    def get_file_blame(
        self,
        repo_url: str,
        file_path: str,
        *,
        branch: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get blame information for a file using GitHub API.

        Returns a list of blame ranges, each containing commit SHA
        and line info.
        """
        normalized = repo_url.strip().rstrip("/")
        if normalized.startswith("https://github.com/"):
            return self._execute_with_rate_limit(
                lambda: self._fetch_github_blame(
                    normalized, file_path, branch
                ),
                name=f"github.blame({normalized}/{file_path})",
            )

        raise ValueError(f"Unsupported git repository host: {repo_url}")

    def get_commit_associated_pr(
        self,
        repo_url: str,
        commit_sha: str,
    ) -> Optional[dict[str, Any]]:
        """Get the PR associated with a specific commit SHA.

        Returns PR data if found, None otherwise.
        """
        normalized = repo_url.strip().rstrip("/")
        if normalized.startswith("https://github.com/"):
            return self._execute_with_rate_limit(
                lambda: self._fetch_github_commit_pr(normalized, commit_sha),
                name=f"github.commit_pr({commit_sha[:7]})",
            )

        raise ValueError(f"Unsupported git repository host: {repo_url}")

    def _fetch_github_blame(
        self,
        repo_url: str,
        file_path: str,
        branch: Optional[str],
        *,
        headers: Optional[dict[str, str]] = None,
    ) -> list[dict[str, Any]]:
        """Fetch blame data from GitHub GraphQL API."""
        parts = repo_url.removeprefix("https://github.com/").split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid GitHub repository URL: {repo_url}")

        owner, repo = parts[0], parts[1]
        if repo.endswith(".git"):
            repo = repo[:-4]

        blame_branch = branch
        if blame_branch is None:
            metadata = self._fetch_github_repo(repo_url)
            blame_branch = (
                metadata.get("default_branch") or "main"
            )

        # Use GitHub GraphQL API for blame
        graphql_url = "https://api.github.com/graphql"

        # GraphQL query for blame
        query = """
        query(
          $owner: String!,
          $repo: String!,
          $branch: String!,
          $path: String!
        ) {
          repository(owner: $owner, name: $repo) {
            ref: object(expression: $branch) {
              ... on Commit {
                blame(path: $path) {
                  ranges {
                    startingLine
                    endingLine
                    commit {
                      oid
                    }
                  }
                }
              }
            }
          }
        }
        """

        variables = {
            "owner": owner,
            "repo": repo,
            "branch": blame_branch,
            "path": file_path,
        }

        payload = {
            "query": query,
            "variables": variables,
        }

        response = self._session.post(
            graphql_url, json=payload, timeout=30, headers=headers
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to retrieve blame for {file_path}: "
                f"{response.status_code}"
            )

        data = response.json()

        # Handle GraphQL errors
        if "errors" in data:
            errors = data.get("errors", [])
            if errors:
                error_msg = errors[0].get("message", "Unknown error")
                # File not found or path doesn't exist
                if (
                    "not found" in error_msg.lower()
                    or "does not exist" in error_msg.lower()
                ):
                    return []
                raise RuntimeError(
                    f"GraphQL error for {file_path}: {error_msg}"
                )

        # Extract blame ranges from GraphQL response
        try:
            ranges = (
                data.get("data", {})
                .get("repository", {})
                .get("ref", {})
                .get("blame", {})
                .get("ranges", [])
            )

            # Convert GraphQL format to match our expected format
            normalized_ranges = []
            for r in ranges:
                normalized_ranges.append({
                    "startingLine": r.get("startingLine"),
                    "endingLine": r.get("endingLine"),
                    "commit": {
                        "sha": r.get("commit", {}).get("oid"),
                    }
                })

            return normalized_ranges
        except (KeyError, AttributeError):
            return []

    def _fetch_github_commit_pr(
        self,
        repo_url: str,
        commit_sha: str,
        *,
        headers: Optional[dict[str, str]] = None,
    ) -> Optional[dict[str, Any]]:
        """Get PR associated with a commit."""
        parts = repo_url.removeprefix("https://github.com/").split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid GitHub repository URL: {repo_url}")

        owner, repo = parts[0], parts[1]
        if repo.endswith(".git"):
            repo = repo[:-4]

        # Use GitHub's commit PR search endpoint
        api_url = (
            f"https://api.github.com/repos/{owner}/{repo}/commits/"
            f"{commit_sha}/pulls"
        )

        response = self._session.get(
            api_url,
            timeout=10,
            headers={
                **({"Accept": "application/vnd.github.groot-preview+json"}),
                **(headers or {}),
            },
        )

        if response.status_code == 404:
            # Commit not found or no associated PR
            return None
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get PR for commit {commit_sha}: "
                f"{response.status_code}"
            )

        prs = response.json()
        if not isinstance(prs, list) or len(prs) == 0:
            return None

        # Return the first (usually only) PR associated with this commit
        return prs[0]

    def _auth_headers(self) -> dict[str, str]:
        """
        _auth_headers: Function description.
        :param:
        :returns:
        """

        token = os.getenv("GITHUB_TOKEN")
        if token:
            return {"Authorization": f"token {token}"}
        return {}
