"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Helpers for handling X-Authorization requirements and tokens.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, MutableMapping, Optional

try:  # pragma: no cover - boto3 available in deployed Lambda
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]

_LOGGER = logging.getLogger(__name__)
_HEADER_KEYS = (
    "X-Authorization",
    "x-authorization",
    "Authorization",
    "authorization",
)

# Defaults align with the spec but are configurable for tests/local runs.
_DDB_TABLE = os.getenv("AUTH_TOKEN_TABLE")
_DEFAULT_EXPIRY_SECONDS = 60 * 60 * 2  # 2 hours
_DEFAULT_MAX_CALLS = 1000


@dataclass
class TokenRecord:
    """
    TokenRecord: Class description.
    """

    token: str
    username: str
    is_admin: bool
    issued_at: float
    usage_count: int = 0
    revoked: bool = False


_TOKENS: MutableMapping[str, TokenRecord] = {}
_DDB_CLIENT = None


def extract_auth_token(
    event: Dict[str, Any],
    *,
    optional: bool | None = None,
    allow_env_default: bool = True,
) -> str | None:
    """Return the provided auth token or raise if authorization is required.

    Legacy helper: inspects headers and optionally falls back to
    AUTH_DEFAULT_TOKEN. The `optional` flag (or AUTH_OPTIONAL env) controls
    whether missing headers raise. The new `require_auth_token` helper below
    should be preferred for strict validation.
    """

    headers = event.get("headers") or {}
    for key in _HEADER_KEYS:
        value = headers.get(key)
        if value:
            return value

    if allow_env_default:
        default_token = os.getenv("AUTH_DEFAULT_TOKEN")
        if default_token:
            return default_token

    env_optional = os.getenv("AUTH_OPTIONAL", "1")
    env_optional_flag = env_optional.lower() not in {"0", "false"}
    optional_flag = env_optional_flag if optional is None else optional

    if optional_flag:
        _LOGGER.debug(
            "X-Authorization header missing; optional bypass active."
        )
        return None

    _LOGGER.warning("Request missing X-Authorization header.")
    raise PermissionError("Missing X-Authorization header")


def issue_token(username: str, *, is_admin: bool) -> str:
    """Create and store a new token for this user."""

    token = f"bearer {uuid.uuid4()}"
    record = TokenRecord(
        token=token,
        username=username,
        is_admin=is_admin,
        issued_at=time.time(),
    )
    if _DDB_TABLE and _ddb_client() is not None:
        _put_token_ddb(record)
    else:
        _TOKENS[token] = record
    return token


def revoke_token(token: str) -> None:
    """Mark an issued token as revoked."""

    record = _TOKENS.get(token)
    if record:
        record.revoked = True


def require_auth_token(
    event: Dict[str, Any],
    *,
    optional: bool | None = None,
    token_store: Optional[MutableMapping[str, TokenRecord]] = None,
    max_calls: int | None = None,
    expiry_seconds: int | None = None,
) -> TokenRecord | None:
    """Validate request authorization and enforce expiry and usage limits.

    Raises PermissionError on missing/invalid/expired/revoked/overused tokens,
    unless optional auth is enabled (flag or AUTH_OPTIONAL env).
    """

    headers = event.get("headers") or {}
    token_value = _extract_header_token(headers)
    env_optional = os.getenv("AUTH_OPTIONAL", "0")
    env_optional_flag = env_optional.lower() not in {"0", "false"}
    optional_flag = env_optional_flag if optional is None else optional

    if not token_value:
        if optional_flag:
            _LOGGER.debug(
                "Auth optional; proceeding without token validation."
            )
            return None
        raise PermissionError("Missing X-Authorization header")

    now = time.time()
    max_age = (
        _get_expiry_seconds() if expiry_seconds is None else expiry_seconds
    )
    max_uses = _get_max_calls() if max_calls is None else max_calls

    if _DDB_TABLE and _ddb_client() is not None and token_store is None:
        updated = _validate_and_increment_usage_ddb(
            token_value,
            max_uses=max_uses,
            max_age=max_age,
        )
        if updated is None:
            raise PermissionError("Invalid authentication token")
        return updated

    try:
        store = token_store if token_store is not None else _TOKENS
        record = store.get(token_value)
    except Exception as exc:  # pragma: no cover - defensive
        _LOGGER.warning("Token lookup failed: %s", exc)
        raise PermissionError("Invalid authentication token") from exc

    if record is None:
        raise PermissionError("Invalid authentication token")

    if record.revoked:
        raise PermissionError("Authentication token has been revoked")
    if now - record.issued_at > max_age:
        raise PermissionError("Authentication token has expired")
    if record.usage_count >= max_uses:
        raise PermissionError("Authentication token usage limit exceeded")

    try:
        record.usage_count += 1
    except PermissionError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        _LOGGER.warning("Token usage increment failed: %s", exc)
        raise PermissionError("Invalid authentication token") from exc
    return record


def _extract_header_token(headers: Dict[str, Any]) -> str | None:
    """
    _extract_header_token: Function description.
    :param headers:
    :returns:
    """

    for key in _HEADER_KEYS:
        value = headers.get(key)
        if value:
            return value
    return None


def _reset_token_store(tokens: Iterable[str] | None = None) -> None:
    """Test-only helper to clear or prune the token store."""

    if tokens is None:
        _TOKENS.clear()
        return
    for token in list(_TOKENS.keys()):
        if token not in tokens:
            _TOKENS.pop(token, None)


def _ddb_client():
    """
    _ddb_client: Function description.
    :param:
    :returns:
    """

    global _DDB_CLIENT
    if _DDB_CLIENT is None:
        if boto3 is None:
            _DDB_CLIENT = None
        else:
            try:
                _DDB_CLIENT = boto3.client("dynamodb")
            except Exception as exc:  # pragma: no cover - environment dependent
                _LOGGER.debug("Failed to initialize DynamoDB client: %s", exc)
                _DDB_CLIENT = None
    return _DDB_CLIENT


def _put_token_ddb(record: TokenRecord) -> None:
    """
    _put_token_ddb: Function description.
    :param record:
    :returns:
    """

    if _DDB_TABLE is None:
        _TOKENS[record.token] = record
        return
    client = _ddb_client()
    if client is None:
        _TOKENS[record.token] = record
        return
    expires_at = int(record.issued_at + _get_expiry_seconds())
    client.put_item(
        TableName=_DDB_TABLE,
        Item={
            "token": {"S": record.token},
            "username": {"S": record.username},
            "is_admin": {"BOOL": record.is_admin},
            "issued_at": {"N": str(int(record.issued_at))},
            "usage_count": {"N": "0"},
            "expires_at": {"N": str(expires_at)},
            "revoked": {"BOOL": False},
        },
    )


def _get_token_ddb(token: str) -> TokenRecord | None:
    """
    _get_token_ddb: Function description.
    :param token:
    :returns:
    """

    if _DDB_TABLE is None:
        return _TOKENS.get(token)
    client = _ddb_client()
    if client is None:
        return _TOKENS.get(token)
    resp = client.get_item(
        TableName=_DDB_TABLE,
        Key={"token": {"S": token}},
        ConsistentRead=True,
    )
    item = resp.get("Item")
    if not item:
        return None
    try:
        return TokenRecord(
            token=token,
            username=item["username"]["S"],
            is_admin=item.get("is_admin", {}).get("BOOL", False),
            issued_at=float(item["issued_at"]["N"]),
            usage_count=int(item.get("usage_count", {}).get("N", "0")),
            revoked=item.get("revoked", {}).get("BOOL", False),
        )
    except Exception:
        return None


def _increment_usage_ddb(token: str, *, max_uses: int, max_age: int) -> None:
    """
    _increment_usage_ddb: Function description.
    :param token:
    :param max_uses:
    :param max_age:
    :returns:
    """

    if _DDB_TABLE is None:
        record = _TOKENS.get(token)
        if record:
            record.usage_count += 1
        return
    client = _ddb_client()
    if client is None:
        record = _TOKENS.get(token)
        if record:
            record.usage_count += 1
        return
    now = int(time.time())
    try:
        client.update_item(
            TableName=_DDB_TABLE,
            Key={"token": {"S": token}},
            UpdateExpression="SET usage_count = usage_count + :inc",
            ConditionExpression=(
                "revoked = :false AND usage_count < :max_uses "
                "AND expires_at > :now"
            ),
            ExpressionAttributeValues={
                ":inc": {"N": "1"},
                ":max_uses": {"N": str(max_uses)},
                ":now": {"N": str(now)},
                ":false": {"BOOL": False},
            },
        )
    except Exception:
        raise PermissionError(
            "Authentication token has expired or exceeded usage"
        )


def _validate_and_increment_usage_ddb(
    token: str,
    *,
    max_uses: int,
    max_age: int,
) -> TokenRecord | None:
    """Validate token constraints and increment usage in a single DDB call."""

    if _DDB_TABLE is None:
        record = _TOKENS.get(token)
        if record:
            record.usage_count += 1
        return record
    client = _ddb_client()
    if client is None:
        record = _TOKENS.get(token)
        if record:
            record.usage_count += 1
        return record

    now = int(time.time())
    min_issued_at = now - max_age
    try:
        client.update_item(
            TableName=_DDB_TABLE,
            Key={"token": {"S": token}},
            UpdateExpression="SET usage_count = usage_count + :inc",
            ConditionExpression=(
                "attribute_exists(#token) "
                "AND (attribute_not_exists(revoked) OR revoked = :false) "
                "AND usage_count < :max_uses "
                "AND issued_at >= :min_issued_at"
            ),
            ExpressionAttributeNames={"#token": "token"},
            ExpressionAttributeValues={
                ":inc": {"N": "1"},
                ":max_uses": {"N": str(max_uses)},
                ":min_issued_at": {"N": str(min_issued_at)},
                ":false": {"BOOL": False},
            },
        )
    except Exception as exc:
        raise PermissionError(
            "Authentication token has expired or exceeded usage"
        ) from exc

    # Most handlers only need validation; avoid fetching the full record to
    # keep per-request latency low (critical for concurrent autograder calls).
    return TokenRecord(
        token=token,
        username="",
        is_admin=False,
        issued_at=float(now),
        usage_count=0,
        revoked=False,
    )


def _get_expiry_seconds() -> int:
    """
    _get_expiry_seconds: Function description.
    :param:
    :returns:
    """

    raw = os.getenv("AUTH_EXPIRY_SECONDS")
    if raw is None:
        return _DEFAULT_EXPIRY_SECONDS
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError("expiry must be positive")
        return value
    except Exception:
        _LOGGER.warning("Invalid AUTH_EXPIRY_SECONDS=%s; using default", raw)
        return _DEFAULT_EXPIRY_SECONDS


def _get_max_calls() -> int:
    """
    _get_max_calls: Function description.
    :param:
    :returns:
    """

    raw = os.getenv("AUTH_MAX_CALLS")
    if raw is None:
        return _DEFAULT_MAX_CALLS
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError("max calls must be positive")
        return value
    except Exception:
        _LOGGER.warning("Invalid AUTH_MAX_CALLS=%s; using default", raw)
        return _DEFAULT_MAX_CALLS
