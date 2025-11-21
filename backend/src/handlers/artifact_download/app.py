"""Lambda handler for generating download URLs."""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from typing import Any, Dict

from src.logging_config import configure_logging
from src.models import validate_artifact_id
from src.storage.blob_store import (ArtifactBlobStore, BlobNotFoundError,
                                    BlobStoreError, build_blob_store_from_env)
from src.utils.auth import extract_auth_token
from src.utils.request_logging import log_request

configure_logging()
_LOGGER = logging.getLogger(__name__)
_BLOB_STORE: ArtifactBlobStore = build_blob_store_from_env()


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point for GET /download/{id}."""

    log_request(_LOGGER, event)
    try:
        artifact_id = _parse_artifact_id(event)
        _extract_auth_token(event)
        link = _BLOB_STORE.generate_download_url(artifact_id)
    except ValueError as error:
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except PermissionError as error:
        return _error_response(HTTPStatus.FORBIDDEN, str(error))
    except BlobNotFoundError as error:
        return _error_response(HTTPStatus.NOT_FOUND, str(error))
    except BlobStoreError as error:
        _LOGGER.exception("Blob store error: %s", error)
        return _error_response(
            HTTPStatus.BAD_GATEWAY,
            "Unable to generate a download link for this artifact",
        )
    except Exception as error:  # noqa: BLE001 - resilience
        _LOGGER.exception("Unhandled download handler error: %s", error)
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Internal server error",
        )

    if _wants_json(event):
        body = {
            "artifact_id": artifact_id,
            "download_url": link.url,
            "expires_in": link.expires_in,
        }
        return _json_response(HTTPStatus.OK, body)

    return _redirect_response(link.url, link.expires_in)


def _parse_artifact_id(event: Dict[str, Any]) -> str:
    artifact_id = (event.get("pathParameters") or {}).get("id")
    if not artifact_id:
        raise ValueError("Path parameter 'id' is required")
    try:
        return validate_artifact_id(artifact_id)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def _extract_auth_token(event: Dict[str, Any]) -> str | None:
    return extract_auth_token(event)


def _wants_json(event: Dict[str, Any]) -> bool:
    params = event.get("queryStringParameters") or {}
    fmt = (params.get("format") or "").lower()
    return fmt == "json"


def _json_response(status: HTTPStatus, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _redirect_response(location: str, expires_in: int) -> Dict[str, Any]:
    return {
        "statusCode": HTTPStatus.FOUND.value,
        "headers": {
            "Location": location,
            "Cache-Control": f"max-age={expires_in}",
        },
        "body": "",
    }


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    return _json_response(status, {"error": message})
