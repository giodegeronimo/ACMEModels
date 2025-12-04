"""Lambda handler for GET /artifacts/{artifact_type}/{id}."""

from __future__ import annotations

import json
import logging
import os
from http import HTTPStatus
from typing import Any, Dict

from src.logging_config import configure_logging
from src.models.artifacts import ArtifactType, validate_artifact_id
from src.storage.errors import ArtifactNotFound
from src.storage.metadata_store import (ArtifactMetadataStore,
                                        build_metadata_store_from_env)
from src.utils.auth import require_auth_token

configure_logging()
_LOGGER = logging.getLogger(__name__)
_METADATA_STORE: ArtifactMetadataStore = build_metadata_store_from_env()


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point for GET /artifacts/{artifact_type}/{id}."""

    try:
        _require_auth(event)
        artifact_type = _parse_artifact_type(event)
        artifact_id = _parse_artifact_id(event)
        artifact = _METADATA_STORE.load(artifact_id)
        if artifact.metadata.type != artifact_type:
            raise ArtifactNotFound(
                f"Artifact '{artifact_id}' not found for type "
                f"'{artifact_type.value}'"
            )
        body = _serialize_artifact(artifact, event)
    except ValueError as error:
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except PermissionError as error:
        return _error_response(HTTPStatus.FORBIDDEN, str(error))
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


def _require_auth(event: Dict[str, Any]) -> None:
    require_auth_token(event, optional=False)


def _serialize_artifact(artifact, event: Dict[str, Any]) -> Dict[str, Any]:
    metadata = artifact.metadata
    data = artifact.data
    download_url = _build_download_endpoint(metadata.id, event)
    return {
        "metadata": {
            "name": metadata.name,
            "id": metadata.id,
            "type": metadata.type.value,
        },
        "data": {
            "url": data.url,
            "download_url": download_url,
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


def _build_download_endpoint(artifact_id: str, event: Dict[str, Any]) -> str:
    """Build the download URL for an artifact."""
    base = _resolve_download_base(event)
    return f"{base}/download/{artifact_id}"


def _resolve_download_base(event: Dict[str, Any]) -> str:
    """Resolve the base URL for download endpoints."""
    base_env = os.environ.get("ARTIFACT_DOWNLOAD_ENDPOINT_BASE")
    if base_env:
        return base_env.rstrip("/")
    context = event.get("requestContext") or {}
    domain = context.get("domainName")
    stage = context.get("stage")
    if domain:
        prefix = f"https://{domain}"
        if stage and stage not in ("", "$default"):
            return f"{prefix}/{stage}"
        return prefix
    return "https://localhost"
