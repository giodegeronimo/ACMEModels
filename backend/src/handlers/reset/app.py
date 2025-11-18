"""Lambda handler for DELETE /reset."""

from __future__ import annotations

import json
import logging
import os
import shutil
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover - boto3 present in AWS, optional locally
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]

from src.storage.blob_store import build_blob_store_from_env
from src.storage.metadata_store import build_metadata_store_from_env

_LOGGER = logging.getLogger(__name__)
_BLOB_STORE = build_blob_store_from_env()
_METADATA_STORE = build_metadata_store_from_env()


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point for DELETE /reset."""

    _extract_auth_token(event)
    try:
        _reset_storage()
    except Exception as error:  # noqa: BLE001
        _LOGGER.exception(
            "Failed to reset registry: %s", error
        )
        return _json_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {"error": "Failed to reset registry"},
        )
    return _json_response(HTTPStatus.OK, {"status": "reset"})


def _extract_auth_token(event: Dict[str, Any]) -> str | None:
    headers = event.get("headers") or {}
    token = headers.get("X-Authorization") or headers.get("x-authorization")
    if not token:
        _LOGGER.info("Registry reset called without X-Authorization header.")
    return token


def _reset_storage() -> None:
    if os.environ.get("AWS_SAM_LOCAL"):
        _reset_local()
        return
    bucket = os.environ.get("ARTIFACT_STORAGE_BUCKET")
    if not bucket:
        _LOGGER.info(
            "No ARTIFACT_STORAGE_BUCKET configured; nothing to reset."
        )
        return
    prefixes = _s3_prefixes()
    _clear_s3_bucket(bucket, prefixes)


def _reset_local() -> None:
    artifact_dir = _local_artifact_dir()
    metadata_dir = _local_metadata_dir()
    for directory in (artifact_dir, metadata_dir):
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)


def _local_artifact_dir() -> Path:
    from src.storage.blob_store import \
        _resolve_local_storage_dir  # type: ignore[attr-defined]

    return _resolve_local_storage_dir()


def _local_metadata_dir() -> Path:
    from src.storage.metadata_store import DEFAULT_METADATA_DIR

    base = Path(os.environ.get("ARTIFACT_METADATA_DIR", DEFAULT_METADATA_DIR))
    base.mkdir(parents=True, exist_ok=True)
    return base


def _s3_prefixes() -> list[str]:
    artifact_prefix = os.environ.get("ARTIFACT_STORAGE_PREFIX", "").strip("/")
    metadata_prefix = os.environ.get(
        "ARTIFACT_METADATA_PREFIX", "metadata"
    ).strip("/")
    prefixes = []
    if artifact_prefix:
        prefixes.append(artifact_prefix)
    if metadata_prefix:
        prefixes.append(metadata_prefix)
    return prefixes


def _clear_s3_bucket(bucket: str, prefixes: list[str]) -> None:
    if boto3 is None:
        raise RuntimeError("boto3 is required to reset the registry in AWS")
    region = (
        os.environ.get("ARTIFACT_STORAGE_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )
    client_kwargs = {"region_name": region} if region else {}
    s3 = boto3.client("s3", **client_kwargs)
    paginator = s3.get_paginator("list_objects_v2")
    to_delete: list[dict[str, str]] = []

    def _flush_batch(batch: list[dict[str, str]]) -> None:
        if not batch:
            return
        s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
        batch.clear()

    for prefix in prefixes or [""]:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                to_delete.append({"Key": obj["Key"]})
                if len(to_delete) == 1000:
                    _flush_batch(to_delete)
        _flush_batch(to_delete)


def _json_response(
    status: HTTPStatus, body: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
