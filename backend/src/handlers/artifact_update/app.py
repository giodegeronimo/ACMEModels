"""Lambda handler for PUT /artifacts/{artifact_type}/{id}."""

from __future__ import annotations

import json
import logging
import os
import shutil
from http import HTTPStatus
from typing import Any, Dict

from src.logging_config import configure_logging
from src.models import Artifact, ArtifactData, ArtifactType
from src.storage.artifact_ingest import (ArtifactBundle, ArtifactDownloadError,
                                         prepare_artifact_bundle)
from src.storage.blob_store import (ArtifactBlobStore, BlobStoreError,
                                    build_blob_store_from_env)
from src.storage.errors import ArtifactNotFound
from src.storage.metadata_store import (ArtifactMetadataStore,
                                        MetadataStoreError,
                                        build_metadata_store_from_env)
from src.utils.auth import require_auth_token

configure_logging()
_LOGGER = logging.getLogger(__name__)
_METADATA_STORE: ArtifactMetadataStore = build_metadata_store_from_env()
_BLOB_STORE: ArtifactBlobStore = build_blob_store_from_env()


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point compatible with AWS Lambda."""

    try:
        _require_auth(event)
        artifact_type = _parse_artifact_type(event)
        artifact_id = _parse_artifact_id(event)
        payload = _parse_body(event)
        incoming_meta = _parse_metadata(payload)
        incoming_data = _parse_data(payload)

        existing = _METADATA_STORE.load(artifact_id)
        _validate_existing(existing, artifact_type, incoming_meta)

        updated_artifact = Artifact(
            metadata=existing.metadata,
            data=ArtifactData(url=incoming_data["url"]),
        )
        _store_artifact_blob(updated_artifact)
        _store_artifact(updated_artifact, replace=True)
        body = _serialize_artifact(updated_artifact, event)
    except ValueError as error:
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except PermissionError as error:
        return _error_response(HTTPStatus.FORBIDDEN, str(error))
    except ArtifactNotFound as error:
        return _error_response(HTTPStatus.NOT_FOUND, str(error))
    except (BlobStoreError, MetadataStoreError) as error:
        return _error_response(HTTPStatus.BAD_GATEWAY, str(error))
    except Exception as error:  # noqa: BLE001
        _LOGGER.exception(
            "Unhandled error in artifact update handler: %s", error
        )
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR, "Internal server error"
        )

    return _json_response(HTTPStatus.OK, body)


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


def _parse_artifact_id(event: Dict[str, Any]) -> str:
    artifact_id = (event.get("pathParameters") or {}).get("id")
    if not artifact_id:
        raise ValueError("Path parameter 'id' is required")
    from src.models.artifacts import validate_artifact_id

    return validate_artifact_id(artifact_id)


def _require_auth(event: Dict[str, Any]) -> None:
    require_auth_token(event, optional=False)


def _parse_body(event: Dict[str, Any]) -> Dict[str, Any]:
    body = event.get("body")
    if body is None:
        raise ValueError("Request body is required")
    if event.get("isBase64Encoded"):
        import base64

        try:
            body = base64.b64decode(body).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("Body could not be decoded from base64") from exc
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
        raise ValueError("Request payload must be an object")
    return payload


def _parse_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Field 'metadata' is required")
    required = {"name", "id", "type"}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"metadata missing fields: {sorted(missing)}")
    return metadata


def _parse_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("Field 'data' is required")
    if "url" not in data:
        raise ValueError("Field 'data.url' is required")
    return data


def _validate_existing(
    existing: Artifact,
    artifact_type: ArtifactType,
    incoming_meta: Dict[str, Any],
) -> None:
    if existing.metadata.type != artifact_type:
        raise ArtifactNotFound(
            f"Artifact '{existing.metadata.id}' not found for type "
            f"'{artifact_type.value}'"
        )
    if incoming_meta.get("id") != existing.metadata.id:
        raise ValueError("metadata.id must match the path parameter")
    if incoming_meta.get("type") != artifact_type.value:
        raise ValueError("metadata.type must match the path artifact_type")
    if incoming_meta.get("name") != existing.metadata.name:
        raise ValueError("metadata.name must match the stored artifact name")


def _store_artifact_blob(artifact: Artifact) -> None:
    bundle: ArtifactBundle | None = None
    try:
        bundle = prepare_artifact_bundle(artifact.data.url)
        if bundle.kind == "file":
            _BLOB_STORE.store_file(
                artifact.metadata.id,
                bundle.path,
                content_type=bundle.content_type,
            )
        else:
            _BLOB_STORE.store_directory(
                artifact.metadata.id,
                bundle.path,
                content_type=bundle.content_type,
            )
    except ArtifactDownloadError as error:
        raise BlobStoreError(str(error)) from error
    finally:
        if bundle:
            if bundle.cleanup_root.is_dir():
                shutil.rmtree(bundle.cleanup_root, ignore_errors=True)
            else:
                bundle.cleanup_root.unlink(missing_ok=True)


def _store_artifact(artifact: Artifact, *, replace: bool = False) -> None:
    _METADATA_STORE.save(artifact, overwrite=replace)


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
