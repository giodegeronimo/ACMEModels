"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Lambda handler for PUT /artifacts/{artifact_type}/{id}.
"""

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
from src.storage.atomic_update import (AtomicUpdateError, AtomicUpdateGroup,
                                       find_exception_in_chain)
from src.storage.blob_store import (ArtifactBlobStore, BlobStoreError,
                                    BlobStoreUnavailableError,
                                    build_blob_store_from_env)
from src.storage.errors import ArtifactNotFound
from src.storage.metadata_store import (ArtifactMetadataStore,
                                        MetadataStoreError,
                                        MetadataStoreUnavailableError,
                                        build_metadata_store_from_env)
from src.utils.auth import require_auth_token

configure_logging()
_LOGGER = logging.getLogger(__name__)
_METADATA_STORE: ArtifactMetadataStore = build_metadata_store_from_env()
_BLOB_STORE: ArtifactBlobStore = build_blob_store_from_env()


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle `PUT /artifacts/{artifact_type}/{id}` updates.

    This endpoint updates the stored artifact's URL (and associated stored blob)
    while requiring that immutable metadata (id/type/name) matches the existing
    record. The update is executed as a best-effort atomic group:

    - Download + store the new artifact blob.
    - Persist the updated metadata record.

    :param event: API Gateway/Lambda proxy event.
    :param context: Lambda context (unused).
    :returns: API Gateway/Lambda proxy response dict.
    """

    try:
        _require_auth(event)
        artifact_type = _parse_artifact_type(event)
        artifact_id = _parse_artifact_id(event)
        payload = _parse_body(event)
        incoming_meta = _parse_metadata(payload)
        incoming_data = _parse_data(payload)

        # Load existing record first so we can enforce immutable metadata
        # invariants and validate the requested artifact_type.
        existing = _METADATA_STORE.load(artifact_id)
        _validate_existing(existing, artifact_type, incoming_meta)

        # Only the data URL is mutable via this endpoint. Metadata is preserved
        # from the stored artifact.
        updated_artifact = Artifact(
            metadata=existing.metadata,
            data=ArtifactData(url=incoming_data["url"]),
        )

        # Write blob + metadata as a coordinated update. The atomic update group
        # provides best-effort rollback/ordering and lets us classify failures.
        group = AtomicUpdateGroup.begin(f"artifact:{artifact_id}")
        group.add_step(
            "store_blob",
            lambda _ctx: _store_artifact_blob(updated_artifact),
        )
        group.add_step(
            "store_metadata",
            lambda _ctx: _store_artifact(updated_artifact, replace=True),
        )
        group.execute()
        body = _serialize_artifact(updated_artifact, event)
    except ValueError as error:
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except PermissionError as error:
        return _error_response(HTTPStatus.FORBIDDEN, str(error))
    except ArtifactNotFound as error:
        return _error_response(HTTPStatus.NOT_FOUND, str(error))
    except (BlobStoreUnavailableError, MetadataStoreUnavailableError) as error:
        _LOGGER.warning("Upstream storage unavailable: %s", error)
        return _error_response(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Storage temporarily unavailable; please retry",
        )
    except AtomicUpdateError as error:
        if find_exception_in_chain(
            error,
            (BlobStoreUnavailableError, MetadataStoreUnavailableError),
        ):
            _LOGGER.warning("Atomic update failed due to outage: %s", error)
            return _error_response(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Storage temporarily unavailable; please retry",
            )
        if find_exception_in_chain(
            error,
            (BlobStoreError, MetadataStoreError),
        ):
            return _error_response(HTTPStatus.BAD_GATEWAY, str(error))
        raise
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
    """
    Parse the `{artifact_type}` path parameter into an `ArtifactType`.

    :param event: API Gateway/Lambda proxy event.
    :returns: Parsed `ArtifactType`.
    :raises ValueError: If the path parameter is missing or invalid.
    """

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
    """
    Parse and validate the `{id}` path parameter.

    :param event: API Gateway/Lambda proxy event.
    :returns: Validated artifact id string.
    :raises ValueError: If the path parameter is missing or invalid.
    """

    artifact_id = (event.get("pathParameters") or {}).get("id")
    if not artifact_id:
        raise ValueError("Path parameter 'id' is required")
    from src.models.artifacts import validate_artifact_id

    return validate_artifact_id(artifact_id)


def _require_auth(event: Dict[str, Any]) -> None:
    """
    Enforce request authentication for this handler.

    :param event: API Gateway/Lambda proxy event.
    :raises PermissionError: If authorization is missing/invalid.
    """

    require_auth_token(event, optional=False)


def _parse_body(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the request JSON body.

    Accepts plain JSON strings, already-decoded dict bodies, and base64-encoded
    payloads when `isBase64Encoded` is set.

    :param event: API Gateway/Lambda proxy event.
    :returns: Parsed request payload dict.
    :raises ValueError: If body is missing, not JSON, or not an object.
    """

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
    """
    Extract and validate the `metadata` section of the request payload.

    :param payload: Parsed request payload dict.
    :returns: `metadata` dict from the request payload.
    :raises ValueError: If metadata is missing or malformed.
    """

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Field 'metadata' is required")
    required = {"name", "id", "type"}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"metadata missing fields: {sorted(missing)}")
    return metadata


def _parse_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and validate the `data` section of the request payload.

    :param payload: Parsed request payload dict.
    :returns: `data` dict from the request payload.
    :raises ValueError: If data is missing or malformed.
    """

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
    """
    Validate that request metadata matches the stored artifact.

    This endpoint treats the artifact's `id`, `type`, and `name` as immutable
    identifiers. The only supported change is updating the `data.url`.

    :param existing: Stored artifact record loaded from metadata storage.
    :param artifact_type: Artifact type from the request path.
    :param incoming_meta: Client-provided `metadata` object from the request.
    :raises ArtifactNotFound: If the stored artifact type doesn't match the path.
    :raises ValueError: If client metadata doesn't match stored metadata.
    """

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
    """
    Download the artifact referenced by `artifact.data.url` and store its blob.

    :param artifact: The artifact whose URL should be downloaded and stored.
    :raises BlobStoreError: If the download fails or blob storage fails.
    """

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
    """
    Persist the artifact record to the metadata store.

    :param artifact: Artifact record to persist.
    :param replace: When True, overwrite the existing record.
    """

    _METADATA_STORE.save(artifact, overwrite=replace)


def _serialize_artifact(
    artifact: Artifact, event: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Serialize the artifact to the public API response payload.

    :param artifact: Artifact to serialize.
    :param event: API Gateway/Lambda proxy event (used to build download URL).
    :returns: JSON-serializable response payload.
    """

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
    """
    Build the download URL returned in the handler response.

    :param artifact_id: Artifact id for the download URL path.
    :param event: API Gateway/Lambda proxy event used to infer base URL.
    :returns: Absolute download URL for this artifact id.
    """

    base = _resolve_download_base(event)
    return f"{base}/download/{artifact_id}"


def _resolve_download_base(event: Dict[str, Any]) -> str:
    """
    Determine the base URL for download links.

    Uses `ARTIFACT_DOWNLOAD_ENDPOINT_BASE` when set, otherwise derives the base
    from the API Gateway request context (domain + stage).

    :param event: API Gateway/Lambda proxy event (request context).
    :returns: Base URL (no trailing slash).
    """

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
    """
    Create a JSON API Gateway proxy response.

    :param status: HTTP status enum to return.
    :param body: JSON-serializable response body.
    :returns: API Gateway/Lambda proxy response dict.
    """

    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    """
    Create a JSON error response payload.

    :param status: HTTP status enum to return.
    :param message: Error message to return under the `error` key.
    :returns: API Gateway/Lambda proxy response dict.
    """

    return _json_response(status, {"error": message})
