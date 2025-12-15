"""

ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.clients.git_client import GitClient
from src.net.rate_limiter import RateLimiter


class DummyLimiter(RateLimiter):
    """
    DummyLimiter: Class description.
    """

    def __init__(self) -> None:
        """
        __init__: Function description.
        :param:
        :returns:
        """

        super().__init__(max_calls=1, period_seconds=1.0)
        self.calls = 0

    def acquire(self) -> None:  # type: ignore[override]
        """
        acquire: Function description.
        :param:
        :returns:
        """

        self.calls += 1


class DummyResponse:
    """
    DummyResponse: Class description.
    """

    def __init__(self, status_code: int, payload: Dict[str, Any]) -> None:
        """
        __init__: Function description.
        :param status_code:
        :param payload:
        :returns:
        """

        self.status_code = status_code
        self._payload = payload

    def json(self) -> Dict[str, Any]:
        """
        json: Function description.
        :param:
        :returns:
        """

        return self._payload


class DummySession:
    """
    DummySession: Class description.
    """

    def __init__(self, response: DummyResponse) -> None:
        """
        __init__: Function description.
        :param response:
        :returns:
        """

        self._response = response
        self.calls: list[tuple[str, int]] = []

    def get(
        self,
        url: str,
        timeout: int,
        headers: dict[str, str] | None = None,
    ) -> DummyResponse:
        """
        get: Function description.
        :param url:
        :param timeout:
        :param headers:
        :returns:
        """

        self.calls.append((url, timeout))
        return self._response

    def post(
        self,
        url: str,
        json: Any,
        timeout: int,
        headers: dict[str, str] | None = None,
    ) -> DummyResponse:
        """
        post: Function description.
        :param url:
        :param json:
        :param timeout:
        :param headers:
        :returns:
        """

        return self._response


class SequenceSession:
    """
    SequenceSession: Class description.
    """

    def __init__(self, responses: List[DummyResponse]) -> None:
        """
        __init__: Function description.
        :param responses:
        :returns:
        """

        self._responses = list(responses)
        self.calls: list[tuple[str, int]] = []

    def get(
        self,
        url: str,
        timeout: int,
        headers: dict[str, str] | None = None,
    ) -> DummyResponse:
        """
        get: Function description.
        :param url:
        :param timeout:
        :param headers:
        :returns:
        """

        self.calls.append((url, timeout))
        if not self._responses:
            raise RuntimeError("No more responses configured")
        return self._responses.pop(0)

    def post(
        self,
        url: str,
        json: Any,
        timeout: int,
        headers: dict[str, str] | None = None,
    ) -> DummyResponse:
        """
        post: Function description.
        :param url:
        :param json:
        :param timeout:
        :param headers:
        :returns:
        """

        if not self._responses:
            raise RuntimeError("No more responses configured")
        return self._responses.pop(0)


def test_git_client_fetches_github_metadata() -> None:
    """
    test_git_client_fetches_github_metadata: Function description.
    :param:
    :returns:
    """

    response = DummyResponse(200, {"full_name": "user/repo"})
    session = DummySession(response)
    limiter = DummyLimiter()
    client = GitClient(rate_limiter=limiter, session=session)

    metadata = client.get_repo_metadata("https://github.com/user/repo")

    assert metadata == {"full_name": "user/repo"}
    assert limiter.calls == 1
    assert session.calls == [
        ("https://api.github.com/repos/user/repo", 10)
    ]


def test_git_client_rejects_unsupported_host() -> None:
    """
    test_git_client_rejects_unsupported_host: Function description.
    :param:
    :returns:
    """

    client = GitClient(rate_limiter=DummyLimiter())

    with pytest.raises(ValueError):
        client.get_repo_metadata("https://gitlab.com/user/repo")


def test_git_client_handles_github_failure() -> None:
    """
    test_git_client_handles_github_failure: Function description.
    :param:
    :returns:
    """

    response = DummyResponse(404, {})
    session = DummySession(response)
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    with pytest.raises(RuntimeError):
        client.get_repo_metadata("https://github.com/user/repo")


class _Response:
    """
    _Response: Class description.
    """

    def __init__(self, status_code: int, payload: Any) -> None:
        """
        __init__: Function description.
        :param status_code:
        :param payload:
        :returns:
        """

        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        """
        json: Function description.
        :param:
        :returns:
        """

        return self._payload


class _MapSession:
    """
    _MapSession: Class description.
    """

    def __init__(
        self,
        *,
        get_map: Optional[dict[str, _Response]] = None,
        post_map: Optional[dict[str, _Response]] = None,
    ) -> None:
        """
        __init__: Function description.
        :param get_map:
        :param post_map:
        :returns:
        """

        self.get_map = get_map or {}
        self.post_map = post_map or {}
        self.get_calls: list[tuple[str, int, Optional[dict[str, str]]]] = []
        self.post_calls: list[tuple[str, int, Any, Optional[dict[str, str]]]] = []

    def get(
        self,
        url: str,
        timeout: int,
        headers: dict[str, str] | None = None,
    ) -> _Response:
        """
        get: Function description.
        :param url:
        :param timeout:
        :param headers:
        :returns:
        """

        self.get_calls.append((url, timeout, headers))
        return self.get_map.get(url, _Response(404, {}))

    def post(
        self,
        url: str,
        json: Any,
        timeout: int,
        headers: dict[str, str] | None = None,
    ) -> _Response:
        """
        post: Function description.
        :param url:
        :param json:
        :param timeout:
        :param headers:
        :returns:
        """

        self.post_calls.append((url, timeout, json, headers))
        return self.post_map.get(url, _Response(500, {}))


def test_list_repo_files_uses_provided_branch() -> None:
    """
    test_list_repo_files_uses_provided_branch: Function description.
    :param:
    :returns:
    """

    session = _MapSession(
        get_map={
            "https://api.github.com/repos/user/repo/git/trees/dev?recursive=1": _Response(
                200,
                {"tree": [{"type": "blob", "path": "a.py"}, {"type": "tree", "path": "dir"}]},
            ),
        }
    )
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    files = client.list_repo_files("https://github.com/user/repo", branch="dev")

    assert files == ["a.py"]


def test_list_repo_files_fetches_default_branch_when_missing() -> None:
    """
    test_list_repo_files_fetches_default_branch_when_missing: Function description.
    :param:
    :returns:
    """

    session = _MapSession(
        get_map={
            "https://api.github.com/repos/user/repo": _Response(200, {"default_branch": "main"}),
            "https://api.github.com/repos/user/repo/git/trees/main?recursive=1": _Response(
                200,
                {"tree": [{"type": "blob", "path": "README.md"}]},
            ),
        }
    )
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    files = client.list_repo_files("https://github.com/user/repo")

    assert files == ["README.md"]


def test_list_repo_contributors_normalizes_and_clamps_per_page() -> None:
    """
    test_list_repo_contributors_normalizes_and_clamps_per_page: Function description.
    :param:
    :returns:
    """

    session = _MapSession(
        get_map={
            "https://api.github.com/repos/user/repo/contributors?per_page=100&anon=1": _Response(
                200,
                [{"login": "alice", "contributions": 1}, {"login": "bob", "contributions": 2}],
            ),
        }
    )
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    contributors = client.list_repo_contributors("https://github.com/user/repo", per_page=1000)

    assert contributors == [
        {"login": "alice", "contributions": 1},
        {"login": "bob", "contributions": 2},
    ]


def test_list_repo_contributors_handles_non_list_payload() -> None:
    """
    test_list_repo_contributors_handles_non_list_payload: Function description.
    :param:
    :returns:
    """

    session = _MapSession(
        get_map={
            "https://api.github.com/repos/user/repo/contributors?per_page=1&anon=1": _Response(
                200,
                {"not": "a list"},
            ),
        }
    )
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    assert client.list_repo_contributors("https://github.com/user/repo", per_page=0) == []


def test_get_file_blame_returns_normalized_ranges() -> None:
    """
    test_get_file_blame_returns_normalized_ranges: Function description.
    :param:
    :returns:
    """

    session = _MapSession(
        get_map={
            "https://api.github.com/repos/user/repo": _Response(200, {"default_branch": "main"}),
        },
        post_map={
            "https://api.github.com/graphql": _Response(
                200,
                {
                    "data": {
                        "repository": {
                            "ref": {
                                "blame": {
                                    "ranges": [
                                        {"startingLine": 1, "endingLine": 2, "commit": {"oid": "abc"}},
                                    ]
                                }
                            }
                        }
                    }
                },
            ),
        },
    )
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    blame = client.get_file_blame("https://github.com/user/repo", "src/app.py")

    assert blame == [
        {"startingLine": 1, "endingLine": 2, "commit": {"sha": "abc"}},
    ]


def test_get_file_blame_handles_graphql_not_found_errors() -> None:
    """
    test_get_file_blame_handles_graphql_not_found_errors: Function description.
    :param:
    :returns:
    """

    session = _MapSession(
        get_map={"https://api.github.com/repos/user/repo": _Response(200, {"default_branch": "main"})},
        post_map={
            "https://api.github.com/graphql": _Response(
                200,
                {"errors": [{"message": "Path does not exist"}]},
            ),
        },
    )
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    assert client.get_file_blame("https://github.com/user/repo", "missing.py") == []


def test_get_file_blame_raises_on_other_graphql_errors() -> None:
    """
    test_get_file_blame_raises_on_other_graphql_errors: Function description.
    :param:
    :returns:
    """

    session = _MapSession(
        get_map={"https://api.github.com/repos/user/repo": _Response(200, {"default_branch": "main"})},
        post_map={
            "https://api.github.com/graphql": _Response(
                200,
                {"errors": [{"message": "boom"}]},
            ),
        },
    )
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    with pytest.raises(RuntimeError, match="GraphQL error"):
        client.get_file_blame("https://github.com/user/repo", "src/app.py")


def test_get_commit_associated_pr_returns_none_on_404() -> None:
    """
    test_get_commit_associated_pr_returns_none_on_404: Function description.
    :param:
    :returns:
    """

    session = _MapSession(
        get_map={
            "https://api.github.com/repos/user/repo/commits/abc/pulls": _Response(404, {}),
        }
    )
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    assert client.get_commit_associated_pr("https://github.com/user/repo", "abc") is None


def test_get_commit_associated_pr_returns_first_pr() -> None:
    """
    test_get_commit_associated_pr_returns_first_pr: Function description.
    :param:
    :returns:
    """

    session = _MapSession(
        get_map={
            "https://api.github.com/repos/user/repo/commits/abc/pulls": _Response(
                200,
                [{"number": 1}, {"number": 2}],
            ),
        }
    )
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    assert client.get_commit_associated_pr("https://github.com/user/repo", "abc") == {"number": 1}


def test_auth_headers_uses_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_auth_headers_uses_token_env: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    client = GitClient(rate_limiter=DummyLimiter(), session=_MapSession())

    assert client._auth_headers() == {"Authorization": "token token"}


def test_auth_headers_empty_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_auth_headers_empty_without_token: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    client = GitClient(rate_limiter=DummyLimiter(), session=_MapSession())

    assert client._auth_headers() == {}


def test_list_repo_files_rejects_unsupported_host() -> None:
    """
    test_list_repo_files_rejects_unsupported_host: Function description.
    :param:
    :returns:
    """

    client = GitClient(rate_limiter=DummyLimiter(), session=_MapSession())
    with pytest.raises(ValueError, match="Unsupported git repository host"):
        client.list_repo_files("https://gitlab.com/user/repo")


def test_list_repo_contributors_rejects_unsupported_host() -> None:
    """
    test_list_repo_contributors_rejects_unsupported_host: Function description.
    :param:
    :returns:
    """

    client = GitClient(rate_limiter=DummyLimiter(), session=_MapSession())
    with pytest.raises(ValueError, match="Unsupported git repository host"):
        client.list_repo_contributors("https://gitlab.com/user/repo")


def test_get_file_blame_rejects_unsupported_host() -> None:
    """
    test_get_file_blame_rejects_unsupported_host: Function description.
    :param:
    :returns:
    """

    client = GitClient(rate_limiter=DummyLimiter(), session=_MapSession())
    with pytest.raises(ValueError, match="Unsupported git repository host"):
        client.get_file_blame("https://gitlab.com/user/repo", "x.py")


def test_get_commit_associated_pr_rejects_unsupported_host() -> None:
    """
    test_get_commit_associated_pr_rejects_unsupported_host: Function description.
    :param:
    :returns:
    """

    client = GitClient(rate_limiter=DummyLimiter(), session=_MapSession())
    with pytest.raises(ValueError, match="Unsupported git repository host"):
        client.get_commit_associated_pr("https://gitlab.com/user/repo", "abc")


def test_invalid_github_repo_url_is_rejected() -> None:
    """
    test_invalid_github_repo_url_is_rejected: Function description.
    :param:
    :returns:
    """

    client = GitClient(rate_limiter=DummyLimiter(), session=_MapSession())
    with pytest.raises(ValueError, match="Invalid GitHub repository URL"):
        client.get_repo_metadata("https://github.com/owner-only")


def test_list_repo_files_raises_on_tree_failure() -> None:
    """
    test_list_repo_files_raises_on_tree_failure: Function description.
    :param:
    :returns:
    """

    session = _MapSession(
        get_map={
            "https://api.github.com/repos/user/repo/git/trees/main?recursive=1": _Response(
                500, {}
            )
        }
    )
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    with pytest.raises(RuntimeError, match="Failed to retrieve repo tree"):
        client.list_repo_files("https://github.com/user/repo", branch="main")


def test_list_repo_contributors_raises_on_api_failure() -> None:
    """
    test_list_repo_contributors_raises_on_api_failure: Function description.
    :param:
    :returns:
    """

    session = _MapSession(
        get_map={
            "https://api.github.com/repos/user/repo/contributors?per_page=10&anon=1": _Response(
                500, {}
            )
        }
    )
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    with pytest.raises(RuntimeError, match="Failed to retrieve repo contributors"):
        client.list_repo_contributors("https://github.com/user/repo", per_page=10)


def test_fetch_blame_malformed_data_returns_empty_list() -> None:
    """
    test_fetch_blame_malformed_data_returns_empty_list: Function description.
    :param:
    :returns:
    """

    session = _MapSession(
        get_map={"https://api.github.com/repos/user/repo": _Response(200, {"default_branch": "main"})},
        post_map={"https://api.github.com/graphql": _Response(200, {"data": None})},
    )
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    assert client.get_file_blame("https://github.com/user/repo", "src/app.py") == []


def test_fetch_blame_raises_on_non_200_response() -> None:
    """
    test_fetch_blame_raises_on_non_200_response: Function description.
    :param:
    :returns:
    """

    session = _MapSession(
        get_map={"https://api.github.com/repos/user/repo": _Response(200, {"default_branch": "main"})},
        post_map={"https://api.github.com/graphql": _Response(500, {})},
    )
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    with pytest.raises(RuntimeError, match="Failed to retrieve blame"):
        client.get_file_blame("https://github.com/user/repo", "src/app.py")


def test_get_commit_associated_pr_handles_non_list_payload() -> None:
    """
    test_get_commit_associated_pr_handles_non_list_payload: Function description.
    :param:
    :returns:
    """

    session = _MapSession(
        get_map={"https://api.github.com/repos/user/repo/commits/abc/pulls": _Response(200, {})}
    )
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    assert client.get_commit_associated_pr("https://github.com/user/repo", "abc") is None


def test_get_commit_associated_pr_raises_on_non_200_response() -> None:
    """
    test_get_commit_associated_pr_raises_on_non_200_response: Function description.
    :param:
    :returns:
    """

    session = _MapSession(
        get_map={"https://api.github.com/repos/user/repo/commits/abc/pulls": _Response(500, {})}
    )
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    with pytest.raises(RuntimeError, match="Failed to get PR for commit"):
        client.get_commit_associated_pr("https://github.com/user/repo", "abc")
