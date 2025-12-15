"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Additional coverage for DynamoDB-backed auth token helpers.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import pytest

from src.utils import auth


@pytest.fixture(autouse=True)
def _reset_auth_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    _reset_auth_state: Function description.
    :param monkeypatch:
    :returns:
    """

    auth._reset_token_store()
    monkeypatch.setattr(auth, "_DDB_TABLE", None)
    monkeypatch.setattr(auth, "_DDB_CLIENT", None)
    monkeypatch.delenv("AUTH_OPTIONAL", raising=False)
    monkeypatch.delenv("AUTH_DEFAULT_TOKEN", raising=False)
    monkeypatch.delenv("AUTH_EXPIRY_SECONDS", raising=False)
    monkeypatch.delenv("AUTH_MAX_CALLS", raising=False)


def test_extract_auth_token_prefers_headers_and_env_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_extract_auth_token_prefers_headers_and_env_default: Function description.
    :param monkeypatch:
    :returns:
    """

    event = {"headers": {"authorization": "bearer header-token"}}
    assert auth.extract_auth_token(event) == "bearer header-token"

    monkeypatch.setenv("AUTH_DEFAULT_TOKEN", "bearer env-token")
    assert auth.extract_auth_token({"headers": {}}, allow_env_default=True) == (
        "bearer env-token"
    )


def test_extract_auth_token_optional_and_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_extract_auth_token_optional_and_strict: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("AUTH_OPTIONAL", "1")
    assert auth.extract_auth_token({"headers": {}}, allow_env_default=False) is None

    monkeypatch.setenv("AUTH_OPTIONAL", "0")
    with pytest.raises(PermissionError):
        auth.extract_auth_token({"headers": {}}, optional=False, allow_env_default=False)


def test_reset_token_store_prunes() -> None:
    """
    test_reset_token_store_prunes: Function description.
    :param:
    :returns:
    """

    token1 = auth.issue_token("alice", is_admin=False)
    token2 = auth.issue_token("bob", is_admin=False)

    auth._reset_token_store(tokens=[token1])

    assert token1 in auth._TOKENS
    assert token2 not in auth._TOKENS


def test_ddb_client_factory_uses_boto3_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_ddb_client_factory_uses_boto3_and_caches: Function description.
    :param monkeypatch:
    :returns:
    """

    sentinel: object = object()
    calls: list[tuple[str]] = []

    class _FakeBoto3:
        """
        _FakeBoto3: Class description.
        """

        def client(self, service: str) -> object:
            """
            client: Function description.
            :param service:
            :returns:
            """

            calls.append((service,))
            return sentinel

    monkeypatch.setattr(auth, "boto3", _FakeBoto3())
    monkeypatch.setattr(auth, "_DDB_CLIENT", None)

    client1 = auth._ddb_client()
    client2 = auth._ddb_client()

    assert client1 is sentinel
    assert client2 is sentinel
    assert calls == [("dynamodb",)]


class _FakeDynamoDB:
    """
    _FakeDynamoDB: Class description.
    """

    def __init__(self) -> None:
        """
        __init__: Function description.
        :param:
        :returns:
        """

        self.items: Dict[str, Dict[str, Dict[str, str]]] = {}
        self.put_calls: int = 0
        self.get_calls: int = 0
        self.update_calls: int = 0

    def put_item(self, *, TableName: str, Item: Dict[str, Dict[str, str]]) -> None:
        """
        put_item: Function description.
        :param TableName:
        :param Item:
        :returns:
        """

        self.put_calls += 1
        token = Item["token"]["S"]
        self.items[token] = dict(Item)

    def get_item(self, *, TableName: str, Key: Dict[str, Dict[str, str]], **_: Any):
        """
        get_item: Function description.
        :param TableName:
        :param Key:
        :param **_:
        :returns:
        """

        self.get_calls += 1
        token = Key["token"]["S"]
        item = self.items.get(token)
        return {"Item": item} if item else {}

    def update_item(
        self,
        *,
        TableName: str,
        Key: Dict[str, Dict[str, str]],
        ExpressionAttributeValues: Dict[str, Dict[str, str]],
        **_: Any,
    ) -> None:
        """
        update_item: Function description.
        :param TableName:
        :param Key:
        :param ExpressionAttributeValues:
        :param **_:
        :returns:
        """

        self.update_calls += 1
        token = Key["token"]["S"]
        item = self.items.get(token)
        if item is None:
            raise RuntimeError("not found")

        now = int(time.time())
        revoked = item.get("revoked", {}).get("BOOL", False)
        usage_count = int(item.get("usage_count", {}).get("N", "0"))
        issued_at = int(item.get("issued_at", {}).get("N", "0"))
        expires_at = int(item.get("expires_at", {}).get("N", "0"))

        max_uses = int(ExpressionAttributeValues.get(":max_uses", {}).get("N", "0"))
        min_issued_at = int(
            ExpressionAttributeValues.get(":min_issued_at", {}).get("N", "0")
        )

        if revoked:
            raise RuntimeError("revoked")
        if max_uses and usage_count >= max_uses:
            raise RuntimeError("too many uses")
        if min_issued_at and issued_at < min_issued_at:
            raise RuntimeError("expired")
        if expires_at <= now:
            raise RuntimeError("expired")

        item["usage_count"] = {"N": str(usage_count + 1)}


def test_issue_token_stores_in_ddb(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_issue_token_stores_in_ddb: Function description.
    :param monkeypatch:
    :returns:
    """

    ddb = _FakeDynamoDB()
    monkeypatch.setattr(auth, "_DDB_TABLE", "tokens")
    monkeypatch.setattr(auth, "_DDB_CLIENT", ddb)

    token = auth.issue_token("alice", is_admin=True)

    assert token.startswith("bearer ")
    assert token not in auth._TOKENS
    assert ddb.put_calls == 1

    record = auth._get_token_ddb(token)
    assert record is not None
    assert record.username == "alice"
    assert record.is_admin is True


def test_get_token_ddb_returns_none_for_invalid_item(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_get_token_ddb_returns_none_for_invalid_item: Function description.
    :param monkeypatch:
    :returns:
    """

    ddb = _FakeDynamoDB()
    monkeypatch.setattr(auth, "_DDB_TABLE", "tokens")
    monkeypatch.setattr(auth, "_DDB_CLIENT", ddb)

    token = "bearer invalid"
    ddb.items[token] = {"token": {"S": token}, "username": {"S": "alice"}}

    assert auth._get_token_ddb(token) is None


def test_validate_and_increment_usage_ddb_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_validate_and_increment_usage_ddb_happy_path: Function description.
    :param monkeypatch:
    :returns:
    """

    ddb = _FakeDynamoDB()
    monkeypatch.setattr(auth, "_DDB_TABLE", "tokens")
    monkeypatch.setattr(auth, "_DDB_CLIENT", ddb)

    token = auth.issue_token("alice", is_admin=False)
    record = auth.require_auth_token(
        {"headers": {"X-Authorization": token}},
        optional=False,
    )
    assert record is not None
    assert ddb.update_calls == 1


def test_require_auth_token_ddb_invalid_token_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_require_auth_token_ddb_invalid_token_branch: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setattr(auth, "_DDB_TABLE", "tokens")
    monkeypatch.setattr(auth, "_DDB_CLIENT", object())

    def fake_validate(*_: Any, **__: Any) -> None:
        """
        fake_validate: Function description.
        :param *_:
        :param **__:
        :returns:
        """

        return None

    monkeypatch.setattr(auth, "_validate_and_increment_usage_ddb", fake_validate)

    with pytest.raises(PermissionError, match="Invalid authentication token"):
        auth.require_auth_token(
            {"headers": {"X-Authorization": "bearer missing"}},
            optional=False,
        )


def test_increment_usage_ddb_raises_when_constraints_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_increment_usage_ddb_raises_when_constraints_fail: Function description.
    :param monkeypatch:
    :returns:
    """

    ddb = _FakeDynamoDB()
    monkeypatch.setattr(auth, "_DDB_TABLE", "tokens")
    monkeypatch.setattr(auth, "_DDB_CLIENT", ddb)

    token = auth.issue_token("alice", is_admin=False)
    item = ddb.items[token]
    item["expires_at"] = {"N": "0"}

    with pytest.raises(PermissionError):
        auth._increment_usage_ddb(token, max_uses=1, max_age=60)


def test_increment_usage_in_memory_when_ddb_disabled() -> None:
    """
    test_increment_usage_in_memory_when_ddb_disabled: Function description.
    :param:
    :returns:
    """

    token = auth.issue_token("alice", is_admin=False)
    auth._increment_usage_ddb(token, max_uses=10, max_age=3600)

    assert auth._TOKENS[token].usage_count == 1


def test_get_expiry_seconds_and_max_calls_invalid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_get_expiry_seconds_and_max_calls_invalid_env: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("AUTH_EXPIRY_SECONDS", "0")
    monkeypatch.setenv("AUTH_MAX_CALLS", "nope")

    assert auth._get_expiry_seconds() == 60 * 60 * 2
    assert auth._get_max_calls() == 1000


class _ExplodingRecord(auth.TokenRecord):
    """
    _ExplodingRecord: Class description.
    """

    _explode: bool = False

    def __setattr__(self, name: str, value: object) -> None:
        """
        __setattr__: Function description.
        :param name:
        :param value:
        :returns:
        """

        if getattr(self, "_explode", False) and name == "usage_count":
            raise PermissionError("usage increment blocked")
        super().__setattr__(name, value)


def test_require_auth_token_re_raises_permission_error_on_increment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_require_auth_token_re_raises_permission_error_on_increment: Function description.
    :param monkeypatch:
    :returns:
    """

    record = _ExplodingRecord(
        token="bearer boom",
        username="alice",
        is_admin=False,
        issued_at=time.time(),
        usage_count=0,
        revoked=False,
    )
    record._explode = True

    with pytest.raises(PermissionError, match="usage increment blocked"):
        auth.require_auth_token(
            {"headers": {"X-Authorization": record.token}},
            optional=False,
            token_store={record.token: record},
        )
