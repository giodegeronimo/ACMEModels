"""Helpers for handling X-Authorization requirements and tokens."""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, MutableMapping, Optional

_LOGGER = logging.getLogger(__name__)
_HEADER_KEYS = (
    "X-Authorization",
    "x-authorization",
    "Authorization",
    "authorization",
)

# Defaults align with the spec but are configurable for tests/local runs.
_DEFAULT_EXPIRY_SECONDS = int(os.getenv("AUTH_EXPIRY_SECONDS", "7200"))
_DEFAULT_MAX_CALLS = int(os.getenv("AUTH_MAX_CALLS", "1000"))


@dataclass
class TokenRecord:
    token: str
    username: str
    is_admin: bool
    issued_at: float
    usage_count: int = 0
    revoked: bool = False


_TOKENS: MutableMapping[str, TokenRecord] = {}


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

    store = token_store if token_store is not None else _TOKENS
    record = store.get(token_value)
    if record is None:
        raise PermissionError("Invalid authentication token")

    now = time.time()
    max_age = (
        _DEFAULT_EXPIRY_SECONDS if expiry_seconds is None else expiry_seconds
    )
    max_uses = _DEFAULT_MAX_CALLS if max_calls is None else max_calls

    if record.revoked:
        raise PermissionError("Authentication token has been revoked")
    if now - record.issued_at > max_age:
        raise PermissionError("Authentication token has expired")
    if record.usage_count >= max_uses:
        raise PermissionError("Authentication token usage limit exceeded")

    record.usage_count += 1
    return record


def _extract_header_token(headers: Dict[str, Any]) -> str | None:
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
