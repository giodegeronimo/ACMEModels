"""Lambda handler for GET /artifacts/{artifact_type}/{id}."""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from typing import Any, Dict

from src.models.artifacts import ArtifactType, validate_artifact_id
from src.storage.errors import ArtifactNotFound
from src.storage.metadata_store import (ArtifactMetadataStore,
                                        build_metadata_store_from_env)

_LOGGER = logging.getLogger(__name__)
_METADATA_STORE: ArtifactMetadataStore = build_metadata_store_from_env()


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point for GET /artifacts/{artifact_type}/{id}."""

    try:
        artifact_type = _parse_artifact_type(event)
        artifact_id = _parse_artifact_id(event)
        _extract_auth_token(event)
        artifact = _METADATA_STORE.load(artifact_id)
        if artifact.metadata.type != artifact_type:
            raise ArtifactNotFound(
                f"Artifact '{artifact_id}' not found for type "
                f"'{artifact_type.value}'"
            )
        body = _serialize_artifact(artifact)
    except ValueError as error:
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except ArtifactNotFound as error:
        return _error_response(HTTPStatus.NOT_FOUND, str(error))
    except Exception as error:  # noqa: BLE001
        _LOGGER.exception("Unhandled error in artifact get handler: %s", error)
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Internal server error",
        )

    return _json_response(HTTPStatus.OK, body)


def _parse_artifact_type(event: Dict[str, Any]) -> ArtifactType:
    raw_type = (event.get("pathParameters") or {}).get("artifact_type")
    if not raw_type:
        raise ValueError("Path parameter 'artifact_type' is required")
    try:
        return ArtifactType(raw_type)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"artifact_type '{raw_type}' is invalid. "
            f"Expected one of {[t.value for t in ArtifactType]}"
        ) from exc


def _parse_artifact_id(event: Dict[str, Any]) -> str:
    artifact_id = (event.get("pathParameters") or {}).get("id")
    if not artifact_id:
        raise ValueError("Path parameter 'id' is required")
    return validate_artifact_id(artifact_id)


def _extract_auth_token(event: Dict[str, Any]) -> str | None:
    headers = event.get("headers") or {}
    token = headers.get("X-Authorization") or headers.get("x-authorization")
    if not token:
        _LOGGER.info("Artifact get called without X-Authorization header.")
    return token


def _serialize_artifact(artifact) -> Dict[str, Any]:
    metadata = artifact.metadata
    data = artifact.data
    return {
        "metadata": {
            "name": metadata.name,
            "id": metadata.id,
            "type": metadata.type.value,
        },
        "data": {
            "url": data.url,
        },
    }


def _json_response(status: HTTPStatus, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    return _json_response(status, {"error": message})
