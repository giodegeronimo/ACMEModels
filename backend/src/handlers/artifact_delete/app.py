"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Lambda handler for DELETE /artifacts/{artifact_type}/{id}.
"""

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
from src.storage.name_index import (NameIndexEntry,
                                    build_name_index_store_from_env)
from src.utils.auth import require_auth_token

try:  # pragma: no cover
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]

configure_logging()
_LOGGER = logging.getLogger(__name__)
_METADATA_STORE: ArtifactMetadataStore = build_metadata_store_from_env()
_NAME_INDEX = build_name_index_store_from_env()
_S3_CLIENT = boto3.client("s3") if boto3 is not None else None


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point for DELETE /artifacts/{artifact_type}/{id}."""

    try:
        _require_auth(event)
        artifact_type = _parse_artifact_type(event)
        artifact_id = _parse_artifact_id(event)

        _LOGGER.info(
            "DELETE request artifact_id=%s type=%s",
            artifact_id,
            artifact_type.value,
        )

        # Verify artifact exists and type matches
        artifact = _METADATA_STORE.load(artifact_id)
        if artifact.metadata.type != artifact_type:
            raise ArtifactNotFound(
                f"Artifact '{artifact_id}' not found for type "
                f"'{artifact_type.value}'"
            )

        # Delete all components of the artifact, collecting errors
        deletion_errors = []

        try:
            _delete_artifact_blob(artifact_id)
        except Exception as exc:
            _LOGGER.warning("Failed to delete artifact blob: %s", exc)
            deletion_errors.append(f"blob: {str(exc)}")

        try:
            _delete_artifact_metadata(artifact_id)
        except Exception as exc:
            _LOGGER.warning("Failed to delete artifact metadata: %s", exc)
            deletion_errors.append(f"metadata: {str(exc)}")

        try:
            _delete_artifact_from_name_index(
                artifact.metadata.name, artifact_id, artifact_type
            )
        except Exception as exc:
            _LOGGER.warning(
                "Failed to delete artifact from name index: %s", exc
            )
            deletion_errors.append(f"name_index: {str(exc)}")

        # Delete rating if it's a model
        if artifact_type == ArtifactType.MODEL:
            try:
                _delete_artifact_rating(artifact_id)
            except Exception as exc:
                _LOGGER.warning("Failed to delete artifact rating: %s", exc)
                deletion_errors.append(f"rating: {str(exc)}")

        if deletion_errors:
            _LOGGER.error(
                "Artifact deletion incomplete for %s. Errors: %s",
                artifact_id,
                "; ".join(deletion_errors),
            )
            return _error_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"Artifact deletion incomplete: {', '.join(deletion_errors)}",
            )

        _LOGGER.info("Successfully deleted artifact %s", artifact_id)
        return _success_response()

    except ValueError as error:
        _LOGGER.warning("Bad request in delete handler: %s", error)
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except PermissionError as error:
        _LOGGER.warning("Permission denied in delete handler: %s", error)
        return _error_response(HTTPStatus.FORBIDDEN, str(error))
    except ArtifactNotFound as error:
        _LOGGER.info("Artifact not found in delete handler: %s", error)
        return _error_response(HTTPStatus.NOT_FOUND, str(error))
    except Exception as error:  # noqa: BLE001
        _LOGGER.exception(
            "Unhandled error in artifact delete handler: %s", error
        )
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Internal server error",
        )


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


def _parse_artifact_id(event: Dict[str, Any]) -> str:
    """Parse and validate `artifact_id` from the request.

    :param event:
    :returns:
    """

    artifact_id = (event.get("pathParameters") or {}).get("id")
    if not artifact_id:
        raise ValueError("Path parameter 'id' is required")
    return validate_artifact_id(artifact_id)


def _require_auth(event: Dict[str, Any]) -> None:
    """Enforce request authentication for this handler.

    :param event:
    :returns:
    """

    require_auth_token(event, optional=False)


def _delete_artifact_blob(artifact_id: str) -> None:
    """Delete artifact binary from S3.

    Raises:
        RuntimeError: If boto3 is not available or bucket is not configured.
        Exception: If S3 deletion fails.
    """
    bucket = os.environ.get("ARTIFACT_STORAGE_BUCKET")
    if not bucket:
        raise RuntimeError("ARTIFACT_STORAGE_BUCKET not configured")

    prefix = os.environ.get("ARTIFACT_STORAGE_PREFIX", "artifacts")
    key = f"{prefix}/{artifact_id}" if prefix else artifact_id

    if _S3_CLIENT is None:
        raise RuntimeError("boto3 is required for deletion")

    _S3_CLIENT.delete_object(Bucket=bucket, Key=key)
    _LOGGER.info("Deleted blob: s3://%s/%s", bucket, key)


def _delete_artifact_metadata(artifact_id: str) -> None:
    """Delete artifact metadata from S3.

    Raises:
        RuntimeError: If boto3 is not available or bucket is not configured.
        Exception: If S3 deletion fails.
    """
    bucket = os.environ.get("ARTIFACT_STORAGE_BUCKET")
    if not bucket:
        raise RuntimeError("ARTIFACT_STORAGE_BUCKET not configured")

    prefix = os.environ.get("ARTIFACT_METADATA_PREFIX", "metadata")
    key = f"{prefix}/{artifact_id}.json" if prefix else f"{artifact_id}.json"

    if _S3_CLIENT is None:
        raise RuntimeError("boto3 is required for deletion")

    _S3_CLIENT.delete_object(Bucket=bucket, Key=key)
    _LOGGER.info("Deleted metadata: s3://%s/%s", bucket, key)


def _delete_artifact_rating(artifact_id: str) -> None:
    """Delete model rating from S3.

    Note: Does not raise if bucket is not configured (ratings are optional).

    Raises:
        RuntimeError: If boto3 is not available.
        Exception: If S3 deletion fails.
    """
    bucket = os.environ.get("MODEL_RESULTS_BUCKET")
    if not bucket:
        _LOGGER.debug("MODEL_RESULTS_BUCKET not configured, skipping")
        return

    prefix = os.environ.get("MODEL_RESULTS_PREFIX", "ratings")
    key = f"{prefix}/{artifact_id}.json" if prefix else f"{artifact_id}.json"

    if _S3_CLIENT is None:
        raise RuntimeError("boto3 is required for deletion")

    _S3_CLIENT.delete_object(Bucket=bucket, Key=key)
    _LOGGER.info("Deleted rating: s3://%s/%s", bucket, key)


def _delete_artifact_from_name_index(
    name: str, artifact_id: str, artifact_type: ArtifactType
) -> None:
    """Delete artifact from DynamoDB name index.

    Raises:
        Exception: If name index deletion fails.
    """
    entry = NameIndexEntry(
        artifact_id=artifact_id,
        name=name,
        artifact_type=artifact_type,
    )
    _NAME_INDEX.delete(entry)
    _LOGGER.info("Deleted from name index: %s", name)


def _success_response() -> Dict[str, Any]:
    """Create an API Gateway proxy response payload.
    :returns:
    """

    return {
        "statusCode": HTTPStatus.OK.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"message": "Artifact deleted successfully"}),
    }


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    """Create a JSON error response payload.

    :param status:
    :param message:
    :returns:
    """

    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": message}),
    }
