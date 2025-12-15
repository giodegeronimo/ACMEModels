"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Unit tests for auth token utilities.
"""

from __future__ import annotations

import time

import pytest

from src.utils import auth


@pytest.fixture(autouse=True)
def _reset_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    _reset_store: Function description.
    :param monkeypatch:
    :returns:
    """

    auth._reset_token_store()
    monkeypatch.delenv("AUTH_OPTIONAL", raising=False)
    monkeypatch.delenv("AUTH_EXPIRY_SECONDS", raising=False)
    monkeypatch.delenv("AUTH_MAX_CALLS", raising=False)


def _event(header_token: str | None) -> dict:
    """
    _event: Function description.
    :param header_token:
    :returns:
    """

    headers = {}
    if header_token is not None:
        headers["X-Authorization"] = header_token
    return {"headers": headers}


def test_require_auth_success_and_usage_increment() -> None:
    """
    test_require_auth_success_and_usage_increment: Function description.
    :param:
    :returns:
    """

    token = auth.issue_token("alice", is_admin=False)
    event = _event(token)

    record = auth.require_auth_token(event, optional=False)
    assert record is not None
    assert record.username == "alice"
    assert record.usage_count == 1

    second = auth.require_auth_token(event, optional=False)
    assert second is not None
    assert record.usage_count == 2


def test_require_auth_missing_header_raises() -> None:
    """
    test_require_auth_missing_header_raises: Function description.
    :param:
    :returns:
    """

    with pytest.raises(PermissionError):
        auth.require_auth_token(_event(None), optional=False)


def test_require_auth_optional_bypass() -> None:
    """
    test_require_auth_optional_bypass: Function description.
    :param:
    :returns:
    """

    record = auth.require_auth_token(_event(None), optional=True)
    assert record is None


def test_require_auth_invalid_token() -> None:
    """
    test_require_auth_invalid_token: Function description.
    :param:
    :returns:
    """

    with pytest.raises(PermissionError):
        auth.require_auth_token(_event("bogus"), optional=False)


def test_require_auth_revoked_token() -> None:
    """
    test_require_auth_revoked_token: Function description.
    :param:
    :returns:
    """

    token = auth.issue_token("bob", is_admin=False)
    auth.revoke_token(token)

    with pytest.raises(PermissionError):
        auth.require_auth_token(_event(token), optional=False)


def test_require_auth_expired_token() -> None:
    """
    test_require_auth_expired_token: Function description.
    :param:
    :returns:
    """

    token = auth.issue_token("carol", is_admin=False)
    record = auth._TOKENS[token]
    record.issued_at = time.time() - 7300  # older than 2h default

    with pytest.raises(PermissionError):
        auth.require_auth_token(_event(token), optional=False)


def test_require_auth_usage_limit_exceeded() -> None:
    """
    test_require_auth_usage_limit_exceeded: Function description.
    :param:
    :returns:
    """

    token = auth.issue_token("dave", is_admin=False)
    event = _event(token)

    auth.require_auth_token(event, optional=False, max_calls=2)
    auth.require_auth_token(event, optional=False, max_calls=2)
    with pytest.raises(PermissionError):
        auth.require_auth_token(event, optional=False, max_calls=2)


def test_env_configurable_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_env_configurable_expiry: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("AUTH_EXPIRY_SECONDS", "1")
    token = auth.issue_token("erin", is_admin=False)
    record = auth._TOKENS[token]
    record.issued_at = time.time() - 2

    with pytest.raises(PermissionError):
        auth.require_auth_token(_event(token), optional=False)


def test_env_configurable_max_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_env_configurable_max_calls: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("AUTH_MAX_CALLS", "1")
    token = auth.issue_token("frank", is_admin=False)
    event = _event(token)

    auth.require_auth_token(event, optional=False)
    with pytest.raises(PermissionError):
        auth.require_auth_token(event, optional=False)


def test_require_auth_token_store_lookup_exception_maps_to_invalid_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_require_auth_token_store_lookup_exception_maps_to_invalid_token: Function description.
    :param monkeypatch:
    :returns:
    """

    class ExplodingStore(dict):
        """
        ExplodingStore: Class description.
        """

        def get(self, key, default=None):  # type: ignore[override]
            """
            get: Function description.
            :param key:
            :param default:
            :returns:
            """

            raise RuntimeError("boom")

    with pytest.raises(PermissionError, match="Invalid authentication token"):
        auth.require_auth_token(
            _event("bearer whatever"),
            optional=False,
            token_store=ExplodingStore(),
        )


def test_require_auth_usage_increment_exception_maps_to_invalid_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_require_auth_usage_increment_exception_maps_to_invalid_token: Function description.
    :param monkeypatch:
    :returns:
    """

    class ExplodingRecord:
        """
        ExplodingRecord: Class description.
        """

        def __init__(self) -> None:
            """
            __init__: Function description.
            :param:
            :returns:
            """

            self.token = "bearer t"
            self.username = "alice"
            self.is_admin = False
            self.issued_at = time.time()
            self.revoked = False

        @property
        def usage_count(self) -> int:
            """
            usage_count: Function description.
            :param:
            :returns:
            """

            return 0

        @usage_count.setter
        def usage_count(self, value: int) -> None:
            """
            usage_count: Function description.
            :param value:
            :returns:
            """

            raise RuntimeError("boom")

    class Store(dict):
        """
        Store: Class description.
        """

        def get(self, key, default=None):  # type: ignore[override]
            """
            get: Function description.
            :param key:
            :param default:
            :returns:
            """

            return ExplodingRecord()

    with pytest.raises(PermissionError, match="Invalid authentication token"):
        auth.require_auth_token(
            _event("bearer t"),
            optional=False,
            token_store=Store(),
        )


def test_validate_and_increment_usage_ddb_failure_maps_to_permission_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_validate_and_increment_usage_ddb_failure_maps_to_permission_error: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setattr(auth, "_DDB_TABLE", "tokens")

    class FakeClient:
        """
        FakeClient: Class description.
        """

        def update_item(self, **kwargs):  # type: ignore[no-untyped-def]
            """
            update_item: Function description.
            :param **kwargs:
            :returns:
            """

            raise RuntimeError("boom")

    monkeypatch.setattr(auth, "_ddb_client", lambda: FakeClient())

    with pytest.raises(
        PermissionError,
        match="Authentication token has expired or exceeded usage",
    ):
        auth.require_auth_token(_event("bearer t"), optional=False)


def test_get_max_calls_invalid_env_triggers_exception_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_get_max_calls_invalid_env_triggers_exception_branch: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("AUTH_MAX_CALLS", "0")
    assert auth._get_max_calls() == 1000
