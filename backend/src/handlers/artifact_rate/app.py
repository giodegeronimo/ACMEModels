"""Lambda handler for GET /artifact/model/{id}/rate."""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from typing import Any, Dict

from src.logging_config import configure_logging
from src.models.artifacts import ArtifactType, validate_artifact_id
from src.storage.errors import ArtifactNotFound
from src.storage.metadata_store import (ArtifactMetadataStore,
                                        build_metadata_store_from_env)
from src.storage.ratings_store import load_rating
from src.utils.auth import extract_auth_token

configure_logging()
_LOGGER = logging.getLogger(__name__)
_METADATA_STORE: ArtifactMetadataStore = build_metadata_store_from_env()


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point for GET /artifact/model/{id}/rate."""

    try:
        artifact_id = _parse_artifact_id(event)
        _extract_auth_token(event)
        artifact = _METADATA_STORE.load(artifact_id)
        if artifact.metadata.type is not ArtifactType.MODEL:
            raise ArtifactNotFound(
                f"Artifact '{artifact_id}' not found for type 'model'"
            )
        rating = load_rating(artifact_id)
        if rating is None:
            raise ArtifactNotFound(
                f"Rating not available for artifact '{artifact_id}'"
            )
        body = rating
    except ValueError as error:
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except PermissionError as error:
        return _error_response(HTTPStatus.FORBIDDEN, str(error))
    except ArtifactNotFound as error:
        return _error_response(HTTPStatus.NOT_FOUND, str(error))
    except Exception as error:  # noqa: BLE001
        _LOGGER.exception(
            "Unhandled error in artifact rate handler: %s", error
        )
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Internal server error",
        )
    return _json_response(HTTPStatus.OK, body)


def _parse_artifact_id(event: Dict[str, Any]) -> str:
    artifact_id = (event.get("pathParameters") or {}).get("id")
    if not artifact_id:
        raise ValueError("Path parameter 'id' is required")
    return validate_artifact_id(artifact_id)


def _extract_auth_token(event: Dict[str, Any]) -> str | None:
    return extract_auth_token(event)


def _json_response(status: HTTPStatus, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    return _json_response(status, {"error": message})
