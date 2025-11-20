"""Helpers for handling X-Authorization requirements."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

_LOGGER = logging.getLogger(__name__)
_HEADER_KEYS = (
    "X-Authorization",
    "x-authorization",
    "Authorization",
    "authorization",
)


def extract_auth_token(
    event: Dict[str, Any],
    *,
    optional: bool | None = None,
) -> str | None:
    """Return the provided auth token or raise if authorization is required."""

    headers = event.get("headers") or {}
    for key in _HEADER_KEYS:
        value = headers.get(key)
        if value:
            return value

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
