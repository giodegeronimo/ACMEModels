"""Shared helpers for logging incoming HTTP requests."""

from __future__ import annotations

import base64
import json
from typing import Any


def log_request(logger, event: Any) -> None:
    """
    Emit a structured log for the current HTTP request.

    Skips non-HTTP events (e.g., async worker payloads) to avoid noise.
    """
    if not isinstance(event, dict):
        return
    request_context = event.get("requestContext") or {}
    http_context = request_context.get("http") or {}
    method = (
        http_context.get("method")
        or event.get("httpMethod")
        or event.get("requestMethod")
    )
    path = (
        http_context.get("path")
        or event.get("path")
        or event.get("rawPath")
        or ""
    )
    if not (method or path or event.get("headers")):
        # Not an API Gateway event.
        return

    query_params = event.get("queryStringParameters") or {}
    path_params = event.get("pathParameters") or {}

    body_repr = _body_preview(event)
    logger.info(
        "HTTP request method=%s path=%s query=%s path_params=%s body=%s",
        method or "<unknown>",
        path,
        query_params,
        path_params,
        body_repr,
    )


def _body_preview(event: dict[str, Any]) -> str:
    if "body" not in event:
        return "<missing>"
    body = event.get("body")
    if body is None:
        return "<null>"
    if isinstance(body, (bytes, bytearray)):
        try:
            decoded = body.decode("utf-8")
        except UnicodeDecodeError:
            return "<binary>"
        return _truncate(decoded)
    if isinstance(body, str):
        if event.get("isBase64Encoded"):
            try:
                decoded = base64.b64decode(body).decode("utf-8")
            except Exception:  # noqa: BLE001
                return "<base64 decode error>"
            return _truncate(decoded)
        return _truncate(body)
    try:
        serialized = json.dumps(body)
    except Exception:  # noqa: BLE001
        serialized = str(body)
    return _truncate(serialized)


def _truncate(value: str, *, limit: int = 2048) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...<truncated>"
