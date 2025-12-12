"""Lambda handler for GET /artifact/model/{id}/rate."""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from http import HTTPStatus
from typing import Any, Dict

from src.logging_config import configure_logging
from src.metrics.ratings import RatingComputationError, compute_model_rating
from src.models import Artifact
from src.models.artifacts import ArtifactType, validate_artifact_id
from src.storage.errors import ArtifactNotFound
from src.storage.metadata_store import (ArtifactMetadataStore,
                                        build_metadata_store_from_env)
from src.storage.ratings_store import (RatingStoreError,
                                       RatingStoreThrottledError,
                                       create_stub_rating, load_rating,
                                       store_rating)
from src.utils.auth import require_auth_token

configure_logging()
_LOGGER = logging.getLogger(__name__)
_METADATA_STORE: ArtifactMetadataStore = build_metadata_store_from_env()
_COMPUTE_TIMEOUT_SECONDS = int(os.getenv("RATING_COMPUTE_TIMEOUT", "240"))
_REQUIRED_RATING_FIELDS = (
    "net_score",
    "ramp_up_time",
    "bus_factor",
    "performance_claims",
    "license",
    "dataset_and_code_score",
    "dataset_quality",
    "code_quality",
    "reproducibility",
    "reviewedness",
    "tree_score",
    "size_score",
)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point for GET /artifact/model/{id}/rate."""

    try:
        _require_auth(event)
        artifact_id = _parse_artifact_id(event)
        _LOGGER.info("Fetching rating for artifact_id=%s", artifact_id)
        artifact = _METADATA_STORE.load(artifact_id)
        if artifact.metadata.type is not ArtifactType.MODEL:
            raise ArtifactNotFound(
                f"Artifact '{artifact_id}' not found for type 'model'"
            )
        rating = _load_rating_with_fallback(artifact)
        body = rating
    except ValueError as error:
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except PermissionError as error:
        return _error_response(HTTPStatus.FORBIDDEN, str(error))
    except ArtifactNotFound as error:
        return _error_response(HTTPStatus.NOT_FOUND, str(error))
    except ServiceUnavailableError as error:
        return _error_response(HTTPStatus.SERVICE_UNAVAILABLE, str(error))
    except Exception as error:  # noqa: BLE001
        _LOGGER.exception(
            "Unhandled error in artifact rate handler: %s", error
        )
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Internal server error",
        )
    return _json_response(HTTPStatus.OK, body)


def _load_rating_with_fallback(artifact: Artifact) -> Dict[str, Any]:
    artifact_id = artifact.metadata.id
    try:
        _LOGGER.info(
            "Attempting to load rating for artifact_id=%s", artifact_id
        )
        rating = load_rating(artifact_id)
    except RatingStoreThrottledError as error:
        _LOGGER.warning(
            "Rating store throttled for artifact_id=%s: %s",
            artifact_id,
            error,
        )
        raise ServiceUnavailableError("Rating service busy") from error
    except RatingStoreError as error:
        _LOGGER.error(
            "Rating store failure for artifact_id=%s: %s",
            artifact_id,
            error,
        )
        raise ServiceUnavailableError("Unable to load rating") from error

    if rating is not None:
        return rating

    _LOGGER.warning(
        "Rating not found for artifact_id=%s. Computing on-demand.",
        artifact_id,
    )
    return _compute_and_store_rating(artifact)


def _compute_and_store_rating(artifact: Artifact) -> Dict[str, Any]:
    artifact_id = artifact.metadata.id
    source_url = artifact.data.url
    start = time.perf_counter()
    try:
        rating = _run_rating_pipeline_with_timeout(source_url)
    except RatingComputationError as error:
        _LOGGER.warning(
            "Rating computation failed for artifact_id=%s: %s. "
            "Returning stub rating.",
            artifact_id,
            error,
        )
        # Return stub instead of raising error
        return create_stub_rating()
    finally:
        elapsed = time.perf_counter() - start
        _LOGGER.info(
            "Rating pipeline finished for artifact_id=%s in %.2fs",
            artifact_id,
            elapsed,
        )

    _validate_rating_payload(rating)
    try:
        store_rating(artifact_id, rating)
    except RatingStoreError as error:
        _LOGGER.warning(
            "Failed to persist rating for artifact_id=%s: %s. "
            "Attempting to read cached copy.",
            artifact_id,
            error,
        )
        try:
            cached = load_rating(artifact_id)
        except RatingStoreError:
            cached = None
        if cached is not None:
            return cached
        raise ServiceUnavailableError(
            "Rating computed but could not be persisted"
        ) from error

    return rating


def _run_rating_pipeline_with_timeout(source_url: str) -> Dict[str, Any]:
    timeout = max(_COMPUTE_TIMEOUT_SECONDS, 1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(compute_model_rating, source_url)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError as error:
            future.cancel()
            raise RatingComputationError(
                f"Timed out after {timeout} seconds"
            ) from error


def _validate_rating_payload(rating: Dict[str, Any]) -> None:
    missing = [
        field for field in _REQUIRED_RATING_FIELDS if field not in rating
    ]
    if missing:
        raise RatingComputationError(
            f"Rating payload missing required fields: {', '.join(missing)}"
        )
    for field in _REQUIRED_RATING_FIELDS:
        value = rating.get(field)
        # size_score is a nested dict with platform-specific scores
        if field == "size_score":
            if not isinstance(value, dict):
                raise RatingComputationError(
                    f"Rating field '{field}' must be a dict"
                )
            # Validate nested numeric values
            for platform, score in value.items():
                if not isinstance(score, (int, float)):
                    raise RatingComputationError(
                        f"Rating field '{field}.{platform}' must be numeric"
                    )
        elif not isinstance(value, (int, float)):
            raise RatingComputationError(
                f"Rating field '{field}' must be numeric"
            )


def _parse_artifact_id(event: Dict[str, Any]) -> str:
    artifact_id = (event.get("pathParameters") or {}).get("id")
    if not artifact_id:
        raise ValueError("Path parameter 'id' is required")
    return validate_artifact_id(artifact_id)


def _require_auth(event: Dict[str, Any]) -> None:
    require_auth_token(event, optional=False)


def _json_response(status: HTTPStatus, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    return _json_response(status, {"error": message})


class ServiceUnavailableError(RuntimeError):
    """Raised when ratings cannot be served right now."""
