"""Lambda handler for GET /tracks."""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from typing import Any, Dict, List

from src.logging_config import configure_logging

configure_logging()
_LOGGER = logging.getLogger(__name__)
_DEFAULT_TRACKS: List[str] = [
    "Performance track",
    "Access control track",
    "High assurance track",
    "Other Security track",
]


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Return the list of planned tracks per the OpenAPI specification."""

    try:
        _log_request(event)
        body = {"plannedTracks": list(_DEFAULT_TRACKS)}
        _LOGGER.info("Tracks response=%s", body)
        return _json_response(HTTPStatus.OK, body)
    except Exception as error:  # noqa: BLE001 - keep handler resilient
        _LOGGER.exception("Unhandled error in tracks handler: %s", error)
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Internal server error",
        )


def _log_request(event: Dict[str, Any]) -> None:
    http_ctx = (event.get("requestContext") or {}).get("http", {})
    _LOGGER.info(
        "Tracks request path=%s headers=%s",
        http_ctx.get("path"),
        event.get("headers"),
    )


def _json_response(status: HTTPStatus, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    return _json_response(status, {"error": message})
