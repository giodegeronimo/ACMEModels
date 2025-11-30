"""Tests for test git client module."""

from __future__ import annotations

from typing import Any, List

import pytest

from src.clients.git_client import GitClient
from src.net.rate_limiter import RateLimiter


class DummyLimiter(RateLimiter):
    def __init__(self) -> None:
        super().__init__(max_calls=1, period_seconds=1.0)
        self.calls = 0

    def acquire(self) -> None:  # type: ignore[override]
        self.calls += 1


class DummyResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class DummySession:
    def __init__(self, response: DummyResponse) -> None:
        self._response = response
        self.calls: list[tuple[str, int]] = []

    def get(
        self,
        url: str,
        timeout: int,
        headers: dict[str, str] | None = None,
    ) -> DummyResponse:
        self.calls.append((url, timeout))
        return self._response

    def post(
        self,
        url: str,
        json: Any,
        timeout: int,
        headers: dict[str, str] | None = None,
    ) -> DummyResponse:
        return self._response


class SequenceSession:
    def __init__(self, responses: List[DummyResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, int]] = []

    def get(
        self,
        url: str,
        timeout: int,
        headers: dict[str, str] | None = None,
    ) -> DummyResponse:
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
        if not self._responses:
            raise RuntimeError("No more responses configured")
        return self._responses.pop(0)


def test_git_client_fetches_github_metadata() -> None:
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
    client = GitClient(rate_limiter=DummyLimiter())

    with pytest.raises(ValueError):
        client.get_repo_metadata("https://gitlab.com/user/repo")


def test_git_client_handles_github_failure() -> None:
    response = DummyResponse(404, {})
    session = DummySession(response)
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    with pytest.raises(RuntimeError):
        client.get_repo_metadata("https://github.com/user/repo")


def test_list_repo_files_uses_default_branch() -> None:
    metadata = DummyResponse(200, {"default_branch": "main"})
    tree = DummyResponse(
        200,
        {
            "tree": [
                {"path": "tests/test_sample.py", "type": "blob"},
                {"path": "README.md", "type": "blob"},
                {"path": "docs", "type": "tree"},
            ]
        },
    )
    session = SequenceSession([metadata, tree])
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    files = client.list_repo_files("https://github.com/user/repo")

    assert files == ["tests/test_sample.py", "README.md"]
    assert len(session.calls) == 2
    assert session.calls[0][0] == "https://api.github.com/repos/user/repo"
    assert session.calls[1][0] == (
        "https://api.github.com/repos/user/repo/git/trees/main"
        "?recursive=1"
    )


def test_list_repo_files_respects_branch() -> None:
    tree = DummyResponse(
        200,
        {"tree": [{"path": "src/app.py", "type": "blob"}]},
    )
    session = SequenceSession([tree])
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    files = client.list_repo_files(
        "https://github.com/user/repo",
        branch="develop",
    )

    assert files == ["src/app.py"]
    assert session.calls == [
        (
            "https://api.github.com/repos/user/repo/git/trees/develop"
            "?recursive=1",
            10,
        )
    ]


def test_list_repo_contributors() -> None:
    contributors = DummyResponse(
        200,
        [
            {"login": "alice", "contributions": 120},
            {"login": "bob", "contributions": 30},
        ],
    )
    session = SequenceSession([contributors])
    client = GitClient(rate_limiter=DummyLimiter(), session=session)

    result = client.list_repo_contributors("https://github.com/org/repo")

    assert result == [
        {"login": "alice", "contributions": 120},
        {"login": "bob", "contributions": 30},
    ]
    assert session.calls == [
        (
            "https://api.github.com/repos/org/repo/contributors"
            "?per_page=100&anon=1",
            10,
        )
    ]
