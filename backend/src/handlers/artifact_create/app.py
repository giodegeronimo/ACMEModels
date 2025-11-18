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
import time
from http import HTTPStatus
from typing import Any, Dict
from urllib.parse import urlparse
from uuid import uuid4

import requests

try:  # pragma: no cover - boto3 present in production
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]

from src.models import Artifact, ArtifactData, ArtifactMetadata, ArtifactType
from src.storage.artifact_ingest import (ArtifactBundle, ArtifactDownloadError,
                                         prepare_artifact_bundle)
from src.storage.blob_store import (ArtifactBlobStore, BlobNotFoundError,
                                    BlobStoreError, StoredArtifact,
                                    build_blob_store_from_env)
from src.storage.errors import ValidationError
from src.storage.metadata_store import (ArtifactMetadataStore,
                                        MetadataStoreError,
                                        build_metadata_store_from_env)
from src.storage.name_index import (build_name_index_store_from_env,
                                    entry_from_metadata)

_LOGGER = logging.getLogger(__name__)
_METADATA_STORE: ArtifactMetadataStore = build_metadata_store_from_env()
_BLOB_STORE: ArtifactBlobStore
_NAME_INDEX = build_name_index_store_from_env()
_LAMBDA_CLIENT = None

ASYNC_TASK_FIELD = "task"
ASYNC_TASK_INGEST = "ingest"


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point compatible with AWS Lambda."""

    if event.get(ASYNC_TASK_FIELD) == ASYNC_TASK_INGEST:
        return _process_async_ingest(event)

    try:
        artifact_type = _parse_artifact_type(event)
        _extract_auth_token(event)  # placeholder – no auth enforcement yet
        payload = _parse_body(event)
        artifact = _build_artifact(artifact_type, payload)
        source_url = payload["url"]
        if _can_process_synchronously(source_url):
            try:
                bundle = prepare_artifact_bundle(source_url)
            except ArtifactDownloadError as error:
                raise BlobStoreError(str(error)) from error
            _store_artifact_blob(artifact, bundle=bundle)
            stored = _store_artifact(
                artifact, readme_excerpt=bundle.readme_excerpt
            )
            response_body = _serialize_artifact(stored, event)
            return _json_response(HTTPStatus.CREATED, response_body)
        _store_artifact(artifact, readme_excerpt=None)
        _enqueue_async_ingest(
            context,
            artifact,
            source_url,
        )
        if _wait_for_download_ready(
            artifact.metadata.id, event, context
        ):
            ready_body = _serialize_artifact(artifact, event)
            return _json_response(HTTPStatus.CREATED, ready_body)
    except ValueError as error:
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except ValidationError as error:
        return _error_response(HTTPStatus.CONFLICT, str(error))
    except BlobStoreError as error:
        return _error_response(HTTPStatus.BAD_GATEWAY, str(error))
    except MetadataStoreError as error:
        return _error_response(HTTPStatus.BAD_GATEWAY, str(error))
    except Exception as error:  # noqa: BLE001 - keep handler resilient
        _LOGGER.exception(
            "Unhandled error in artifact create handler: %s", error
        )
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR, "Internal server error"
        )

    response_body = _serialize_artifact(artifact, event)
    response_body["status"] = "processing"
    return _json_response(HTTPStatus.ACCEPTED, response_body)


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
    repo_id = _resolve_hf_repo_id(parsed)
    if repo_id:
        _, repo = repo_id.split("/", 1)
        return repo
    candidate = parsed.path.rstrip("/").split("/")[-1] if parsed.path else ""
    if not candidate:
        candidate = parsed.netloc or "artifact"
    return candidate


def _resolve_hf_repo_id(parsed_url) -> str | None:
    if not (parsed_url.netloc or "").endswith("huggingface.co"):
        return None
    segments = [segment for segment in parsed_url.path.split("/") if segment]
    if "resolve" not in segments:
        return None
    resolve_index = segments.index("resolve")
    candidates = segments[:resolve_index]
    while candidates and candidates[0] in {"datasets", "spaces", "models"}:
        candidates = candidates[1:]
    if len(candidates) >= 2:
        owner, repo = candidates[:2]
        return f"{owner}/{repo}"
    return None


def _extract_github_repo(parsed_url) -> tuple[str, str] | None:
    netloc = (parsed_url.netloc or "").lower()
    if not netloc.endswith("github.com"):
        return None
    segments = [segment for segment in parsed_url.path.split("/") if segment]
    if len(segments) < 2:
        return None
    owner, repo = segments[:2]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


def _generate_artifact_id() -> str:
    # UUID4 hex matches the allowed `[a-zA-Z0-9-]+` regex
    # while remaining unique.
    return uuid4().hex


def _store_artifact_blob(
    artifact: Artifact,
    *,
    bundle: ArtifactBundle,
) -> StoredArtifact:
    try:
        if bundle.kind == "file":
            stored = _BLOB_STORE.store_file(
                artifact.metadata.id,
                bundle.path,
                content_type=bundle.content_type,
            )
        else:
            stored = _BLOB_STORE.store_directory(
                artifact.metadata.id,
                bundle.path,
                content_type=bundle.content_type,
            )
    finally:
        if bundle.cleanup_root.is_dir():
            shutil.rmtree(bundle.cleanup_root, ignore_errors=True)
        else:
            bundle.cleanup_root.unlink(missing_ok=True)
    return stored


def _store_artifact(
    artifact: Artifact,
    *,
    replace: bool = False,
    readme_excerpt: str | None = None,
) -> Artifact:
    _METADATA_STORE.save(artifact, overwrite=replace)
    try:
        _NAME_INDEX.save(
            entry_from_metadata(
                artifact.metadata,
                readme_excerpt=readme_excerpt,
            )
        )
    except Exception as exc:  # noqa: BLE001 - keep ingest resilient
        _LOGGER.warning(
            "Failed to update name index for %s: %s",
            artifact.metadata.id,
            exc,
        )
    return artifact


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


def _lambda_client():
    global _LAMBDA_CLIENT
    if _LAMBDA_CLIENT is None:
        if boto3 is None:  # pragma: no cover - handled in tests via patching
            raise BlobStoreError("boto3 is required for async ingest")
        _LAMBDA_CLIENT = boto3.client("lambda")
    return _LAMBDA_CLIENT


def _can_process_synchronously(source_url: str) -> bool:
    parsed = urlparse(source_url)
    host = parsed.netloc or ""
    path = parsed.path or ""
    if host.endswith("huggingface.co") and "/resolve/" not in path:
        # Hugging Face repo roots require enumerating and downloading many
        # files, which easily exceeds API Gateway's sync timeout. Handle them
        # async.
        return False
    try:
        response = requests.head(source_url, allow_redirects=True, timeout=5)
        response.raise_for_status()
        size = int(response.headers.get("Content-Length", "0"))
    except Exception:  # noqa: BLE001 - fall back to async
        return False
    max_size_bytes = int(
        os.environ.get("SYNC_INGEST_MAX_BYTES", str(25 * 1024 * 1024))
    )
    return 0 < size <= max_size_bytes


def _enqueue_async_ingest(
    context: Any,
    artifact: Artifact,
    source_url: str,
) -> None:
    function_arn = getattr(context, "invoked_function_arn", None)
    if not function_arn:
        function_arn = os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    if not function_arn:
        raise BlobStoreError("Unable to determine worker function name")

    payload = {
        ASYNC_TASK_FIELD: ASYNC_TASK_INGEST,
        "artifact": {
            "metadata": {
                "name": artifact.metadata.name,
                "id": artifact.metadata.id,
                "type": artifact.metadata.type.value,
            }
        },
        "source_url": source_url,
    }

    if os.environ.get("ACME_DISABLE_ASYNC") == "1":
        _process_async_ingest(payload)
        return

    try:
        _lambda_client().invoke(
            FunctionName=function_arn,
            InvocationType="Event",
            Payload=json.dumps(payload).encode("utf-8"),
        )
    except Exception as exc:  # noqa: BLE001
        raise BlobStoreError("Failed to enqueue artifact ingest task") from exc


def _process_async_ingest(event: Dict[str, Any]) -> Dict[str, Any]:
    artifact_payload = event.get("artifact", {})
    metadata_payload = artifact_payload.get("metadata", {})
    source_url = event.get("source_url")
    if not source_url:
        _LOGGER.error("Async ingest missing source_url")
        return {"statusCode": 400, "body": "missing source_url"}

    try:
        metadata = ArtifactMetadata(
            name=metadata_payload["name"],
            id=metadata_payload["id"],
            type=ArtifactType(metadata_payload["type"]),
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception(
            "Invalid artifact metadata in async payload: %s", exc
        )
        return {"statusCode": 400, "body": "invalid payload"}

    artifact = Artifact(metadata=metadata, data=ArtifactData(url=source_url))
    try:
        bundle = prepare_artifact_bundle(source_url)
    except ArtifactDownloadError as error:
        _LOGGER.exception(
            "Async bundle preparation failed for %s",
            metadata.id,
        )
        raise BlobStoreError(str(error)) from error
    try:
        _store_artifact_blob(artifact, bundle=bundle)
        _store_artifact(
            artifact,
            replace=True,
            readme_excerpt=bundle.readme_excerpt,
        )
        _LOGGER.info("Async ingest done for artifact %s", metadata.id)
    except Exception as error:  # noqa: BLE001
        _LOGGER.exception(
            "Async ingest failed for %s: %s", metadata.id, error
        )
        raise
    return {"statusCode": 200, "body": "ok"}


def _wait_for_download_ready(
    artifact_id: str, event: Dict[str, Any], context: Any
) -> bool:
    if os.environ.get("ACME_DISABLE_ASYNC") == "1":
        return True

    remaining_ms = 0
    if hasattr(context, "get_remaining_time_in_millis"):
        remaining_ms = context.get_remaining_time_in_millis()
    max_wait = min(
        float(os.environ.get("SYNC_INGEST_MAX_WAIT", "25")),
        max(0.0, remaining_ms / 1000.0 - 2.0),
    )
    if max_wait <= 0:
        return False
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        try:
            _BLOB_STORE.generate_download_url(artifact_id)
            return True
        except BlobNotFoundError:
            time.sleep(0.5)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("Download readiness check failed: %s", exc)
            time.sleep(0.5)
    return False
