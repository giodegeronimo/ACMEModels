from __future__ import annotations

from typing import Any, Dict

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
    def __init__(self, status_code: int, payload: Dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Dict[str, Any]:
        return self._payload


class DummySession:
    def __init__(self, response: DummyResponse) -> None:
        self._response = response
        self.calls: list[tuple[str, int]] = []

    def get(self, url: str, timeout: int) -> DummyResponse:
        self.calls.append((url, timeout))
        return self._response


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
