"""Persistent storage helpers for model ratings."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]
    Config = None  # type: ignore[assignment]

    class ClientError(Exception):  # type: ignore[no-redef]
        """Placeholder when botocore is unavailable."""

    class BotoCoreError(Exception):  # type: ignore[no-redef]
        """Placeholder when botocore is unavailable."""

_LOCAL_RATINGS_DIR = Path(
    os.environ.get("ARTIFACT_RATINGS_DIR", "/tmp/acme-artifact-ratings")
)
_LOGGER = logging.getLogger(__name__)
_S3_CLIENT = None


class RatingStoreError(RuntimeError):
    """Raised when rating persistence fails."""


class RatingStoreThrottledError(RatingStoreError):
    """Raised when S3 throttles rating operations."""


def _build_s3_client() -> Any:
    global _S3_CLIENT
    if _S3_CLIENT is not None:
        return _S3_CLIENT

    if boto3 is None:  # pragma: no cover - boto3 always present in prod
        raise RatingStoreError("boto3 is required for rating storage")

    region = (
        os.environ.get("ARTIFACT_STORAGE_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )
    client_kwargs: Dict[str, Any] = {}

    if region:
        client_kwargs["region_name"] = region

    if Config is not None:
        cfg = Config(
            retries={"max_attempts": 5, "mode": "standard"},
        )
        client_kwargs["config"] = cfg

    _S3_CLIENT = boto3.client("s3", **client_kwargs)
    return _S3_CLIENT


def store_rating(artifact_id: str, rating: Dict[str, Any]) -> None:
    if os.environ.get("AWS_SAM_LOCAL"):
        _LOCAL_RATINGS_DIR.mkdir(parents=True, exist_ok=True)
        path = _LOCAL_RATINGS_DIR / f"{artifact_id}.json"
        path.write_text(json.dumps(rating), encoding="utf-8")
        return
    bucket = os.environ.get("MODEL_RESULTS_BUCKET")
    if not bucket:
        raise RuntimeError("MODEL_RESULTS_BUCKET is not configured")
    prefix = os.environ.get("MODEL_RESULTS_PREFIX", "ratings").strip("/")
    key = f"{prefix}/{artifact_id}.json" if prefix else f"{artifact_id}.json"
    client = _build_s3_client()
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(rating).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as exc:  # pragma: no cover - network errors
        _LOGGER.error(
            "Failed to store rating for artifact_id=%s: %s",
            artifact_id,
            exc,
        )
        raise RatingStoreError(
            f"Failed to store rating for '{artifact_id}'"
        ) from exc


def load_rating(artifact_id: str) -> Dict[str, Any] | None:
    """Load rating from S3 or local filesystem.

    Returns None for any error to enable on-demand computation fallback.
    """
    if os.environ.get("AWS_SAM_LOCAL"):
        path = _LOCAL_RATINGS_DIR / f"{artifact_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    bucket = os.environ.get("MODEL_RESULTS_BUCKET")
    if not bucket:
        return None
    prefix = os.environ.get("MODEL_RESULTS_PREFIX", "ratings").strip("/")
    key = f"{prefix}/{artifact_id}.json" if prefix else f"{artifact_id}.json"
    client = _build_s3_client()
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read())
    except Exception as exc:  # pragma: no cover - network errors
        # Return None for any error to allow on-demand computation
        _LOGGER.debug(
            "Could not load rating for artifact_id=%s: %s",
            artifact_id,
            exc,
        )
        return None
