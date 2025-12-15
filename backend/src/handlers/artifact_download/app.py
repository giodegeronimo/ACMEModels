"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Lambda handler for generating download URLs.
"""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from typing import Any, Dict

from src.logging_config import configure_logging
from src.models import validate_artifact_id
from src.storage.blob_store import (ArtifactBlobStore, BlobNotFoundError,
                                    BlobStoreError, BlobStoreUnavailableError,
                                    build_blob_store_from_env)
from src.utils.auth import require_auth_token

configure_logging()
_LOGGER = logging.getLogger(__name__)
_BLOB_STORE: ArtifactBlobStore = build_blob_store_from_env()


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point for GET /download/{id}."""

    try:
        _require_auth(event)
        artifact_id = _parse_artifact_id(event)
        link = _BLOB_STORE.generate_download_url(artifact_id)
    except ValueError as error:
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except PermissionError as error:
        return _error_response(HTTPStatus.FORBIDDEN, str(error))
    except BlobNotFoundError as error:
        return _error_response(HTTPStatus.NOT_FOUND, str(error))
    except BlobStoreUnavailableError as error:
        _LOGGER.warning("Blob store temporarily unavailable: %s", error)
        return _error_response(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Storage temporarily unavailable; please retry",
        )
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
    """Parse and validate `artifact_id` from the request.

    :param event:
    :returns:
    """

    artifact_id = (event.get("pathParameters") or {}).get("id")
    if not artifact_id:
        raise ValueError("Path parameter 'id' is required")
    try:
        return validate_artifact_id(artifact_id)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def _require_auth(event: Dict[str, Any]) -> None:
    """Enforce request authentication for this handler.

    :param event:
    :returns:
    """

    require_auth_token(event, optional=False)


def _wants_json(event: Dict[str, Any]) -> bool:
    """Helper function.

    :param event:
    :returns:
    """

    params = event.get("queryStringParameters") or {}
    fmt = (params.get("format") or "").lower()
    return fmt == "json"


def _json_response(status: HTTPStatus, body: Dict[str, Any]) -> Dict[str, Any]:
    """Create a JSON API Gateway proxy response.

    :param status:
    :param body:
    :returns:
    """

    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _redirect_response(location: str, expires_in: int) -> Dict[str, Any]:
    """Create an API Gateway proxy response payload.

    :param location:
    :param expires_in:
    :returns:
    """

    return {
        "statusCode": HTTPStatus.FOUND.value,
        "headers": {
            "Location": location,
            "Cache-Control": f"max-age={expires_in}",
        },
        "body": "",
    }


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    """Create a JSON error response payload.

    :param status:
    :param message:
    :returns:
    """

    return _json_response(status, {"error": message})
