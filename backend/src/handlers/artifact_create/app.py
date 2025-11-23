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

import tarfile
import zipfile
from pathlib import Path

from src.logging_config import configure_logging
from src.metrics.ratings import RatingComputationError, compute_model_rating
from src.models import Artifact, ArtifactData, ArtifactMetadata, ArtifactType
from src.storage.artifact_ingest import (ArtifactBundle, ArtifactDownloadError,
                                         prepare_artifact_bundle)
from src.storage.blob_store import (ArtifactBlobStore, BlobNotFoundError,
                                    BlobStoreError, StoredArtifact,
                                    build_blob_store_from_env)
from src.storage.errors import ValidationError
from src.storage.lineage_store import (LineageStore,
                                       build_lineage_store_from_env)
from src.storage.metadata_store import (ArtifactMetadataStore,
                                        MetadataStoreError,
                                        build_metadata_store_from_env)
from src.storage.name_index import (build_name_index_store_from_env,
                                    entry_from_metadata)
from src.storage.ratings_store import store_rating
from src.utils.auth import extract_auth_token

configure_logging()
_LOGGER = logging.getLogger(__name__)
_METADATA_STORE: ArtifactMetadataStore = build_metadata_store_from_env()
_BLOB_STORE: ArtifactBlobStore
_NAME_INDEX = build_name_index_store_from_env()
_LINEAGE_STORE: LineageStore = build_lineage_store_from_env()
_LAMBDA_CLIENT = None

ASYNC_TASK_FIELD = "task"
ASYNC_TASK_INGEST = "ingest"


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point compatible with AWS Lambda."""

    if event.get(ASYNC_TASK_FIELD) == ASYNC_TASK_INGEST:
        _LOGGER.info(
            "Processing async ingest payload for artifact_id=%s",
            (event.get("artifact") or {}).get("metadata", {}).get("id"),
        )
        return _process_async_ingest(event)

    try:
        artifact_type = _parse_artifact_type(event)
        _extract_auth_token(event)
        payload = _parse_body(event)
        artifact = _build_artifact(artifact_type, payload)
        source_url = payload["url"]
        _LOGGER.info(
            "artifact_id=%s type=%s name=%s url=%s",
            artifact.metadata.id,
            artifact_type.value,
            artifact.metadata.name,
            source_url,
        )
        if _can_process_synchronously(source_url):
            _LOGGER.info(
                "Processing artifact_id=%s synchronously",
                artifact.metadata.id,
            )
            try:
                bundle = prepare_artifact_bundle(source_url)
            except ArtifactDownloadError as error:
                raise BlobStoreError(str(error)) from error
            # Attempt to extract lineage metadata from the downloaded bundle
            _extract_and_upsert_lineage(bundle, artifact)
            _store_artifact_blob(artifact, bundle=bundle)
            _compute_and_store_rating_if_needed(artifact, source_url)
            stored = _store_artifact(
                artifact, readme_excerpt=bundle.readme_excerpt
            )
            _LOGGER.info(
                "artifact_id=%s stored synchronously",
                stored.metadata.id,
            )
            response_body = _serialize_artifact(stored, event)
            return _json_response(HTTPStatus.CREATED, response_body)
        _LOGGER.info(
            "artifact_id=%s requires async ingest",
            artifact.metadata.id,
        )
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
        _log_bad_request(event, error)
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except PermissionError as error:
        return _error_response(HTTPStatus.FORBIDDEN, str(error))
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
    return extract_auth_token(event)


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

    url_value = parsed["url"]
    if not isinstance(url_value, str) or not url_value.strip():
        raise ValueError("Field 'url' must be a non-empty string")

    result: Dict[str, Any] = {"url": url_value.strip()}
    if "name" in parsed and parsed["name"] is not None:
        name_value = parsed["name"]
        if not isinstance(name_value, str) or not name_value.strip():
            raise ValueError("Field 'name' must be a non-empty string")
        result["name"] = name_value.strip()
    return result


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
    provided_name = payload.get("name")
    metadata = ArtifactMetadata(
        name=provided_name or _derive_artifact_name(url),
        id=_generate_artifact_id(),
        type=artifact_type,
    )
    data = ArtifactData(url=url)
    return Artifact(metadata=metadata, data=data)


def _compute_and_store_rating_if_needed(
    artifact: Artifact, source_url: str
) -> None:
    if artifact.metadata.type is not ArtifactType.MODEL:
        return
    _LOGGER.info(
        "Computing rating for artifact_id=%s",
        artifact.metadata.id,
    )
    try:
        rating_payload = compute_model_rating(source_url)
    except RatingComputationError as exc:
        raise ValidationError(str(exc)) from exc
    store_rating(artifact.metadata.id, rating_payload)


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
    _LOGGER.debug(
        "Storing blob artifact_id=%s kind=%s path=%s",
        artifact.metadata.id,
        bundle.kind,
        bundle.path,
    )
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
    _LOGGER.info(
        "Persisting metadata/index artifact_id=%s replace=%s",
        artifact.metadata.id,
        replace,
    )
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
        _LOGGER.debug("Creating boto3 lambda client for async ingest")
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
        _LOGGER.info(
            "Invoking async ingest for artifact_id=%s function=%s",
            artifact.metadata.id,
            function_arn,
        )
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
    _LOGGER.info("Async ingest start artifact_id=%s", metadata.id)
    try:
        bundle = prepare_artifact_bundle(source_url)
    except ArtifactDownloadError as error:
        _LOGGER.exception(
            "Async bundle preparation failed for %s",
            metadata.id,
        )
        raise BlobStoreError(str(error)) from error
    # Try to extract lineage from the bundle for async ingest as well.
    _extract_and_upsert_lineage(bundle, artifact)
    try:
        _store_artifact_blob(artifact, bundle=bundle)
        _compute_and_store_rating_if_needed(artifact, source_url)
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
    _LOGGER.info(
        "Waiting up to %.1fs for artifact_id=%s to finish ingest",
        max_wait,
        artifact_id,
    )
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        try:
            _BLOB_STORE.generate_download_url(artifact_id)
            _LOGGER.info("Artifact_id=%s ready during sync wait", artifact_id)
            return True
        except BlobNotFoundError:
            time.sleep(0.5)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("Download readiness check failed: %s", exc)
            time.sleep(0.5)
    _LOGGER.info("Artifact_id=%s not ready before timeout", artifact_id)
    return False


def _log_bad_request(event: Dict[str, Any], error: Exception) -> None:
    try:
        body = event.get("body")
        if event.get("isBase64Encoded") and isinstance(body, str):
            import base64

            body = base64.b64decode(body).decode("utf-8")
    except Exception:
        body = "<unable to decode body>"
    try:
        path = (event.get("requestContext") or {}).get("http", {}).get("path")
    except Exception:
        path = None
    path_display = path or event.get("path") or ""
    _LOGGER.warning(
        "Bad request for artifact_create path=%s body=%s error=%s",
        path_display,
        body,
        error,
    )


def _read_file_from_bundle(
    bundle: ArtifactBundle,
    filename: str,
) -> str | None:
    """Try to locate and read a file with `filename` inside the bundle.

    Returns the file contents as a string if found, otherwise None.
    """
    try:
        if bundle.kind == "directory":
            root = Path(bundle.path)
            for candidate in root.rglob(filename):
                if candidate.is_file():
                    return candidate.read_text(encoding="utf-8")
            return None

        # bundle.kind == 'file'
        path = Path(bundle.path)
        # Try tar.gz / tar archives
        try:
            if tarfile.is_tarfile(path):
                with tarfile.open(path) as archive:
                    for member in archive.getmembers():
                        if (
                            member.isfile()
                            and Path(member.name).name == filename
                        ):
                            f = archive.extractfile(member)
                            if f is None:
                                continue
                            data = f.read()
                            return data.decode("utf-8")
        except Exception:
            # fall through to zip handling
            pass

        # Try zip archives
        try:
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as archive:
                    for info in archive.infolist():
                        if (
                            not info.is_dir()
                            and Path(info.filename).name == filename
                        ):
                            with archive.open(info) as handle:
                                data = handle.read()
                                return data.decode("utf-8")
        except Exception:
            pass

        # As a last resort, if the file itself is JSON, try to read it.
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None
    except Exception as exc:  # noqa: BLE001
        # do not fail ingest on lineage issues
        _LOGGER.debug("Error while searching bundle for %s: %s", filename, exc)
        return None


def _extract_and_upsert_lineage(
    bundle: ArtifactBundle,
    artifact: Artifact,
) -> None:
    """Extract lineage JSON from the bundle (if present) and upsert it.

    This function is best-effort: any failure will be logged and will not
    interrupt the ingest flow.
    """
    artifact_id = artifact.metadata.id
    try:
        # Look for common filenames
        content = _read_file_from_bundle(bundle, "lineage.json")
        if content is None:
            content = _read_file_from_bundle(bundle, "config.json")
        if content is None:
            return
        # Parse JSON and construct domain objects
        payload = json.loads(content)
        nodes_raw = payload.get("nodes", [])
        edges_raw = payload.get("edges", [])
        from src.models.lineage import (ArtifactLineageEdge,
                                        ArtifactLineageGraph,
                                        ArtifactLineageNode)

        nodes = [ArtifactLineageNode(**n) for n in nodes_raw]
        edges = [ArtifactLineageEdge(**e) for e in edges_raw]
        graph = ArtifactLineageGraph(nodes=nodes, edges=edges)
        # Persist lineage for other handlers
        try:
            _LINEAGE_STORE.save(  # type: ignore[attr-defined]
                artifact_id, graph
            )
            _LOGGER.info("Lineage persisted for artifact=%s", artifact_id)
        except Exception as exc:  # noqa: BLE001 - do not fail ingest
            _LOGGER.warning(
                "Failed to persist lineage for %s: %s",
                artifact_id,
                exc,
            )

        # Upsert into the in-memory lineage repo if available (tests/dev)
        try:
            from src.storage.memory import get_lineage_repo

            repo = get_lineage_repo()
            repo.upsert(artifact_id, graph)
            _LOGGER.info("Lineage upserted for artifact=%s", artifact_id)
        except Exception as exc:  # noqa: BLE001 - do not fail ingest
            _LOGGER.warning(
                "Failed to upsert lineage for %s: %s",
                artifact_id,
                exc,
            )
    except Exception as exc:  # noqa: BLE001 - swallow and log
        _LOGGER.warning(
            "Failed to extract lineage from bundle for %s: %s",
            artifact_id,
            exc,
        )
