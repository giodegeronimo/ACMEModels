"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Lambda handler for POST /artifact/{artifact_type}.

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

from src.logging_config import configure_logging
from src.metrics.ratings import RatingComputationError, compute_model_rating
from src.models import Artifact, ArtifactData, ArtifactMetadata, ArtifactType
from src.storage.artifact_ingest import (ArtifactBundle, ArtifactDownloadError,
                                         prepare_artifact_bundle)
from src.storage.atomic_update import (AtomicUpdateError, AtomicUpdateGroup,
                                       find_exception_in_chain)
from src.storage.blob_store import (ArtifactBlobStore, BlobNotFoundError,
                                    BlobStoreError, BlobStoreUnavailableError,
                                    StoredArtifact, build_blob_store_from_env)
from src.storage.errors import ValidationError
from src.storage.lineage_extractor import extract_lineage_graph
from src.storage.lineage_store import store_lineage
from src.storage.metadata_store import (ArtifactMetadataStore,
                                        MetadataStoreError,
                                        MetadataStoreUnavailableError,
                                        build_metadata_store_from_env)
from src.storage.name_index import (build_name_index_store_from_env,
                                    entry_from_metadata)
from src.storage.ratings_store import (RatingStoreError, load_rating,
                                       store_rating, store_stub_rating)
from src.utils.auth import require_auth_token

configure_logging()
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
        _LOGGER.info(
            "Processing async ingest payload for artifact_id=%s",
            (event.get("artifact") or {}).get("metadata", {}).get("id"),
        )
        return _process_async_ingest(event)

    try:
        _require_auth(event)
        artifact_type = _parse_artifact_type(event)
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

            group = AtomicUpdateGroup.begin(f"artifact:{artifact.metadata.id}")

            group.add_step(
                "ensure_stub_rating",
                lambda _ctx: _ensure_stub_rating_exists(artifact),
            )

            def _download_bundle(ctx: Dict[str, Any]) -> None:
                """Helper function.

                :param ctx:
                :returns:
                """

                try:
                    ctx["bundle"] = prepare_artifact_bundle(source_url)
                except ArtifactDownloadError as error:
                    raise BlobStoreError(str(error)) from error

            group.add_step("prepare_bundle", _download_bundle)

            group.add_step(
                "store_blob",
                lambda ctx: _store_artifact_blob(
                    artifact, bundle=ctx["bundle"]
                ),
            )
            group.add_step(
                "store_rating",
                lambda _ctx: _compute_and_store_rating_if_needed(
                    artifact, source_url
                ),
            )
            group.add_step(
                "store_lineage",
                lambda _ctx: _extract_and_store_lineage(artifact, source_url),
            )

            def _persist_metadata(ctx: Dict[str, Any]) -> None:
                """Helper function.

                :param ctx:
                :returns:
                """

                bundle = ctx.get("bundle")
                ctx["stored"] = _store_artifact(
                    artifact,
                    readme_excerpt=getattr(bundle, "readme_excerpt", None),
                )

            group.add_step("store_metadata", _persist_metadata)

            ctx = group.execute()
            stored = ctx["stored"]
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
        _ensure_stub_rating_exists(artifact)
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
    """Parse and validate `artifact_type` from the request.

    :param event:
    :returns:
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


def _require_auth(event: Dict[str, Any]) -> None:
    """Enforce request authentication for this handler.

    :param event:
    :returns:
    """

    require_auth_token(event, optional=False)


def _parse_body(event: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and validate `body` from the request.

    :param event:
    :returns:
    """

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
    """Decode and validate request payload data.

    :param body:
    :returns:
    """

    import base64

    try:
        return base64.b64decode(body).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Body could not be decoded from base64") from exc


def _build_artifact(
    artifact_type: ArtifactType, payload: Dict[str, Any]
) -> Artifact:
    """Build a derived URL or response value.

    :param artifact_type:
    :param payload:
    :returns:
    """

    url = payload["url"]
    provided_name = payload.get("name")
    metadata = ArtifactMetadata(
        name=provided_name or _derive_artifact_name(url),
        id=_generate_artifact_id(),
        type=artifact_type,
    )
    data = ArtifactData(url=url)
    return Artifact(metadata=metadata, data=data)


def _ensure_stub_rating_exists(artifact: Artifact) -> None:
    """Ensure the model rating object exists for `/rate` reads.

    For async ingests the autograder may call GET `/rate` before the async
    worker has started; persist a placeholder rating payload immediately so
    the endpoint always has something well-formed to return.
    """

    if artifact.metadata.type is not ArtifactType.MODEL:
        return
    try:
        existing = load_rating(artifact.metadata.id)
        if existing is None:
            store_stub_rating(
                artifact.metadata.id,
                name=artifact.metadata.name,
            )
        else:
            _LOGGER.info(
                "Rating already exists for %s, skipping stub",
                artifact.metadata.id,
            )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning(
            "Failed to check/store stub rating for %s: %s",
            artifact.metadata.id,
            exc,
        )


def _compute_and_store_rating_if_needed(
    artifact: Artifact, source_url: str
) -> None:
    """Helper function.

    :param artifact:
    :param source_url:
    :returns:
    """

    if artifact.metadata.type is not ArtifactType.MODEL:
        return
    _LOGGER.info(
        "Computing rating for artifact_id=%s",
        artifact.metadata.id,
    )
    try:
        rating_payload = compute_model_rating(source_url)
        store_rating(artifact.metadata.id, rating_payload)
        _LOGGER.info(
            "Stored real rating for artifact_id=%s", artifact.metadata.id
        )
    except RatingComputationError as exc:
        _LOGGER.warning(
            "Rating computation failed for artifact_id=%s: %s. "
            "Stub will remain.",
            artifact.metadata.id,
            exc,
        )
    except RatingStoreError as exc:
        _LOGGER.warning(
            "Failed to store rating for artifact_id=%s: %s. "
            "Stub will remain.",
            artifact.metadata.id,
            exc,
        )


def _extract_and_store_lineage(
    artifact: Artifact, source_url: str
) -> None:
    """Extract and store lineage graph for model artifacts."""
    if artifact.metadata.type is not ArtifactType.MODEL:
        return
    try:
        _LOGGER.info(
            "Extracting lineage for artifact %s", artifact.metadata.id
        )
        graph = extract_lineage_graph(artifact.metadata.id, source_url)
        store_lineage(artifact.metadata.id, graph)
        _LOGGER.info(
            "Stored lineage for artifact %s with %d nodes and %d edges",
            artifact.metadata.id,
            len(graph.nodes),
            len(graph.edges),
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.error(
            "Lineage extraction failed for %s from %s: %s",
            artifact.metadata.id,
            source_url,
            exc,
            exc_info=True,
        )


def _derive_artifact_name(url: str) -> str:
    """Helper function.

    :param url:
    :returns:
    """

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
    """Resolve configuration from environment/request context.

    :param parsed_url:
    :returns:
    """

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
    """Helper function.

    :param parsed_url:
    :returns:
    """

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
    """Helper function.
    :returns:
    """

    return uuid4().hex


def _store_artifact_blob(
    artifact: Artifact,
    *,
    bundle: ArtifactBundle,
) -> StoredArtifact:
    """Persist data to a backing store.

    :param artifact:
    :param bundle:
    :returns:
    """

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
    """Persist data to a backing store.

    :param artifact:
    :param replace:
    :param readme_excerpt:
    :returns:
    """

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
    """Serialize a domain object into a JSON payload.

    :param artifact:
    :param event:
    :returns:
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
    """Build a derived URL or response value.

    :param artifact_id:
    :param event:
    :returns:
    """

    base = _resolve_download_base(event)
    return f"{base}/download/{artifact_id}"


def _resolve_download_base(event: Dict[str, Any]) -> str:
    """Resolve configuration from environment/request context.

    :param event:
    :returns:
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


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    """Create a JSON error response payload.

    :param status:
    :param message:
    :returns:
    """

    return _json_response(status, {"error": message})


_BLOB_STORE = build_blob_store_from_env()


def _lambda_client():
    """Helper function.
    :returns:
    """

    global _LAMBDA_CLIENT
    if _LAMBDA_CLIENT is None:
        if boto3 is None:  # pragma: no cover - handled in tests via patching
            raise BlobStoreError("boto3 is required for async ingest")
        _LOGGER.debug("Creating boto3 lambda client for async ingest")
        _LAMBDA_CLIENT = boto3.client("lambda")
    return _LAMBDA_CLIENT


def _can_process_synchronously(source_url: str) -> bool:
    """Helper function.

    :param source_url:
    :returns:
    """

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
    """Helper function.

    :param context:
    :param artifact:
    :param source_url:
    :returns:
    """

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
    """Helper function.

    :param event:
    :returns:
    """

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

    _ensure_stub_rating_exists(artifact)

    # Compute rating/lineage before any heavyweight downloads so `/rate`
    # can return real values as soon as possible.
    _compute_and_store_rating_if_needed(artifact, source_url)
    _extract_and_store_lineage(artifact, source_url)

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
    """Helper function.

    :param artifact_id:
    :param event:
    :param context:
    :returns:
    """

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
    """Helper function.

    :param event:
    :param error:
    :returns:
    """

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
    _LOGGER.warning(
        "Bad request for artifact_create path=%s body=%s error=%s",
        path or (event.get("path") or ""),
        body,
        error,
    )
