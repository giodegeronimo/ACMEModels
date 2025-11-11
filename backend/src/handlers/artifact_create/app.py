"""Lambda handler for POST /artifact/{artifact_type}.

This implementation wires the HTTP request to the in-memory storage adapter.
Authentication/authorization is currently a placeholder – the handler merely
extracts the token so that we can hook it in later without changing the
function signature.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse
from uuid import uuid4

from src.models import Artifact, ArtifactData, ArtifactMetadata, ArtifactType
from src.storage.artifact_ingest import (ArtifactDownloadError,
                                         prepare_artifact_bundle)
from src.storage.blob_store import (ArtifactBlobStore, BlobStoreError,
                                    StoredArtifact, build_blob_store_from_env)
from src.storage.errors import RepositoryError, ValidationError
from src.storage.memory import InMemoryArtifactRepository

_LOGGER = logging.getLogger(__name__)
_REPO = InMemoryArtifactRepository()
_BLOB_STORE: ArtifactBlobStore


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point compatible with AWS Lambda."""

    try:
        artifact_type = _parse_artifact_type(event)
        _extract_auth_token(event)  # placeholder – no auth enforcement yet
        payload = _parse_body(event)
        artifact = _build_artifact(artifact_type, payload)
        _store_artifact_blob(artifact)
        stored = _store_artifact(artifact)
    except ValueError as error:
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except ValidationError as error:
        # Repository-level validation is only triggered for duplicates.
        return _error_response(HTTPStatus.CONFLICT, str(error))
    except BlobStoreError as error:
        return _error_response(HTTPStatus.BAD_GATEWAY, str(error))
    except RepositoryError as error:
        _LOGGER.exception("Storage error during artifact create: %s", error)
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Failed to persist the artifact",
        )
    except Exception as error:  # noqa: BLE001 - keep handler resilient
        _LOGGER.exception(
            "Unhandled error in artifact create handler: %s", error
        )
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR, "Internal server error"
        )

    response_body = _serialize_artifact(stored, event)
    return _json_response(HTTPStatus.CREATED, response_body)


def _parse_artifact_type(event: Dict[str, Any]) -> ArtifactType:
    raw_type = (event.get("pathParameters") or {}).get("artifact_type")
    if not raw_type:
        raise ValueError("Path parameter 'artifact_type' is required")
    try:
        return ArtifactType(raw_type)
    except ValueError as exc:
        raise ValueError(
            f"artifact_type '{raw_type}' is invalid. "
            f"Expected one of {[t.value for t in ArtifactType]}"
        ) from exc


def _extract_auth_token(event: Dict[str, Any]) -> str | None:
    headers = event.get("headers") or {}
    token = headers.get("X-Authorization") or headers.get("x-authorization")
    if not token:
        _LOGGER.info("Artifact create called without X-Authorization header.")
    return token


def _parse_body(event: Dict[str, Any]) -> Dict[str, Any]:
    body = event.get("body")
    if body is None:
        raise ValueError("Request body is required")
    if event.get("isBase64Encoded"):
        body = _decode_base64_body(body)
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON") from exc
    elif isinstance(body, dict):
        parsed = body
    else:
        raise ValueError("Request body type is not supported")

    if not isinstance(parsed, dict):
        raise ValueError("Request body must be a JSON object")
    if "url" not in parsed:
        raise ValueError("Field 'url' is required")
    if len(parsed) != 1:
        raise ValueError("Only the 'url' field is supported in this request")
    return parsed


def _decode_base64_body(body: str) -> str:
    import base64

    try:
        return base64.b64decode(body).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Body could not be decoded from base64") from exc


def _build_artifact(
    artifact_type: ArtifactType, payload: Dict[str, Any]
) -> Artifact:
    url = payload["url"]
    metadata = ArtifactMetadata(
        name=_derive_artifact_name(url),
        id=_generate_artifact_id(),
        type=artifact_type,
    )
    data = ArtifactData(url=url)
    return Artifact(metadata=metadata, data=data)


def _derive_artifact_name(url: str) -> str:
    parsed = urlparse(url)
    candidate = parsed.path.rstrip("/").split("/")[-1] if parsed.path else ""
    if not candidate:
        candidate = parsed.netloc or "artifact"
    return candidate


def _generate_artifact_id() -> str:
    # UUID4 hex matches the allowed `[a-zA-Z0-9-]+` regex
    # while remaining unique.
    return uuid4().hex


def _store_artifact_blob(artifact: Artifact) -> StoredArtifact:
    file_path: Path | None = None
    try:
        file_path, content_type = prepare_artifact_bundle(artifact.data.url)
        stored = _BLOB_STORE.store_file(
            artifact.metadata.id,
            file_path,
            content_type=content_type,
        )
    except ArtifactDownloadError as error:
        raise BlobStoreError(str(error)) from error
    finally:
        if file_path:
            file_path.unlink(missing_ok=True)
            parent = file_path.parent
            # Clean up temp directories created for HF archives.
            if parent.name.startswith("hf_repo_"):
                shutil.rmtree(parent, ignore_errors=True)
    return stored


def _store_artifact(artifact: Artifact) -> Artifact:
    return _REPO.create(artifact)


def _serialize_artifact(
    artifact: Artifact, event: Dict[str, Any]
) -> Dict[str, Any]:
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


def _build_download_endpoint(
    artifact_id: str, event: Dict[str, Any]
) -> str:
    base = _resolve_download_base(event)
    return f"{base}/download/{artifact_id}"


def _resolve_download_base(event: Dict[str, Any]) -> str:
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


def _json_response(status: HTTPStatus, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    return _json_response(status, {"error": message})


_BLOB_STORE = build_blob_store_from_env()
