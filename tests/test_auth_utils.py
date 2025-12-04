"""Unit tests for auth token utilities."""

from __future__ import annotations

import time

import pytest

from src.utils import auth


@pytest.fixture(autouse=True)
def _reset_store(monkeypatch: pytest.MonkeyPatch) -> None:
    auth._reset_token_store()
    monkeypatch.delenv("AUTH_OPTIONAL", raising=False)
    monkeypatch.delenv("AUTH_EXPIRY_SECONDS", raising=False)
    monkeypatch.delenv("AUTH_MAX_CALLS", raising=False)


def _event(header_token: str | None) -> dict:
    headers = {}
    if header_token is not None:
        headers["X-Authorization"] = header_token
    return {"headers": headers}


def test_require_auth_success_and_usage_increment() -> None:
    token = auth.issue_token("alice", is_admin=False)
    event = _event(token)

    record = auth.require_auth_token(event, optional=False)
    assert record.username == "alice"
    assert record.usage_count == 1

    auth.require_auth_token(event, optional=False)
    assert record.usage_count == 2


def test_require_auth_missing_header_raises() -> None:
    with pytest.raises(PermissionError):
        auth.require_auth_token(_event(None), optional=False)


def test_require_auth_optional_bypass() -> None:
    record = auth.require_auth_token(_event(None), optional=True)
    assert record is None


def test_require_auth_invalid_token() -> None:
    with pytest.raises(PermissionError):
        auth.require_auth_token(_event("bogus"), optional=False)


def test_require_auth_revoked_token() -> None:
    token = auth.issue_token("bob", is_admin=False)
    auth.revoke_token(token)

    with pytest.raises(PermissionError):
        auth.require_auth_token(_event(token), optional=False)


def test_require_auth_expired_token() -> None:
    token = auth.issue_token("carol", is_admin=False)
    record = auth._TOKENS[token]
    record.issued_at = time.time() - 7300  # older than 2h default

    with pytest.raises(PermissionError):
        auth.require_auth_token(_event(token), optional=False)


def test_require_auth_usage_limit_exceeded() -> None:
    token = auth.issue_token("dave", is_admin=False)
    event = _event(token)

    auth.require_auth_token(event, optional=False, max_calls=2)
    auth.require_auth_token(event, optional=False, max_calls=2)
    with pytest.raises(PermissionError):
        auth.require_auth_token(event, optional=False, max_calls=2)
