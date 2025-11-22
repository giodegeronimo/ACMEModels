"""Lambda handler for PUT /authenticate."""

from __future__ import annotations

import json
import logging
import os
from http import HTTPStatus
from typing import Any, Dict

from src.logging_config import configure_logging

configure_logging()
_LOGGER = logging.getLogger(__name__)
_DEFAULT_USERNAME = os.environ.get(
    "DEFAULT_ADMIN_USERNAME", "ece30861defaultadminuser"
)
_DEFAULT_PASSWORDS = {
    os.environ.get(
        "DEFAULT_ADMIN_PASSWORD",
        "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE artifacts;",
    ),
    os.environ.get(
        "ALT_ADMIN_PASSWORD",
        "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;",
    ),
}
_TOKEN_VALUE = os.environ.get(
    "DEFAULT_AUTH_TOKEN",
    "bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.default",
)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Authenticate the default admin user and issue a token."""

    try:
        _log_request(event)
        payload = _parse_body(event)
        _validate_credentials(payload)
        _LOGGER.info("Authentication success for user=%s", _DEFAULT_USERNAME)
        return _json_response(HTTPStatus.OK, _TOKEN_VALUE)
    except ValueError as error:
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except PermissionError as error:
        return _error_response(HTTPStatus.UNAUTHORIZED, str(error))
    except Exception as error:  # noqa: BLE001
        _LOGGER.exception("Unhandled error in authenticate handler: %s", error)
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR, "Internal server error"
        )


def _parse_body(event: Dict[str, Any]) -> Dict[str, Any]:
    body = event.get("body")
    if body is None:
        raise ValueError("Request body is required")
    if isinstance(body, str):
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON") from exc
    elif isinstance(body, dict):
        payload = body
    else:
        raise ValueError("Request body type is not supported")
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    return payload


def _validate_credentials(payload: Dict[str, Any]) -> None:
    user = payload.get("user")
    secret = payload.get("secret")
    if not isinstance(user, dict) or not isinstance(secret, dict):
        raise ValueError("Fields 'user' and 'secret' are required")
    username = user.get("name")
    password = secret.get("password")
    if not isinstance(username, str) or not isinstance(password, str):
        raise ValueError(
            "Fields 'user.name' and 'secret.password' must be strings"
        )
    if username != _DEFAULT_USERNAME:
        _LOGGER.warning("Authentication failed for unknown user=%s", username)
        raise PermissionError("Invalid credentials")
    if password not in _DEFAULT_PASSWORDS:
        _LOGGER.warning(
            "Authentication failed for user=%s due to bad password",
            username,
        )
        raise PermissionError("Invalid credentials")


def _json_response(status: HTTPStatus, body: Any) -> Dict[str, Any]:
    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    return _json_response(status, {"error": message})


def _log_request(event: Dict[str, Any]) -> None:
    http_ctx = (event.get("requestContext") or {}).get("http", {})
    _LOGGER.info(
        "Auth request path=%s headers=%s",
        http_ctx.get("path"),
        event.get("headers"),
    )
