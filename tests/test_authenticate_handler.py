"""Tests for PUT /authenticate handler."""

from __future__ import annotations

import json
from typing import Any, Dict

from backend.src.handlers.authenticate import app as handler
from src.utils import auth

_PRIMARY_PASSWORD = (
    "correcthorsebatterystaple123(!__+@**(A'\"`;"
    "DROP TABLE artifacts;"
)
_ALTERNATE_PASSWORD = (
    "correcthorsebatterystaple123(!__+@**(A'\"`;"
    "DROP TABLE packages;"
)
_EXPECTED_TOKEN = "bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.default"


def _event(body: Any | None) -> Dict[str, Any]:
    return {
        "requestContext": {"http": {"method": "PUT", "path": "/authenticate"}},
        "headers": {},
        "body": json.dumps(body) if body is not None else None,
    }


def _payload(password: str) -> Dict[str, Any]:
    return {
        "user": {"name": "ece30861defaultadminuser", "is_admin": True},
        "secret": {"password": password},
    }


def test_authenticate_success_with_primary_password() -> None:
    auth._reset_token_store()
    response = handler.lambda_handler(
        _event(_payload(_PRIMARY_PASSWORD)),
        None,
    )

    assert response["statusCode"] == 200
    token = json.loads(response["body"])
    # Token should be registered in the in-memory store and non-empty
    assert isinstance(token, str) and token.startswith("bearer ")
    assert auth._TOKENS[token].username == "ece30861defaultadminuser"


def test_authenticate_success_with_alternate_password() -> None:
    auth._reset_token_store()
    response = handler.lambda_handler(
        _event(_payload(_ALTERNATE_PASSWORD)),
        None,
    )

    assert response["statusCode"] == 200
    token = json.loads(response["body"])
    assert isinstance(token, str) and token.startswith("bearer ")


def test_authenticate_rejects_bad_username() -> None:
    body = {
        "user": {"name": "nope", "is_admin": True},
        "secret": {"password": _PRIMARY_PASSWORD},
    }
    response = handler.lambda_handler(_event(body), None)

    assert response["statusCode"] == 401
    assert json.loads(response["body"]) == {"error": "Invalid credentials"}


def test_authenticate_requires_body() -> None:
    response = handler.lambda_handler(_event(None), None)

    assert response["statusCode"] == 400
    assert json.loads(response["body"]) == {
        "error": "Request body is required"
    }


def test_authenticate_rejects_invalid_json() -> None:
    event = {
        "requestContext": {
            "http": {"method": "PUT", "path": "/authenticate"},
        },
        "headers": {},
        "body": "{not-json}",
    }
    response = handler.lambda_handler(event, None)

    assert response["statusCode"] == 400
    assert json.loads(response["body"]) == {
        "error": "Request body must be valid JSON"
    }
