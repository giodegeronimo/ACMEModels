"""Persistent storage helpers for model ratings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]

_LOCAL_RATINGS_DIR = Path(
    os.environ.get("ARTIFACT_RATINGS_DIR", "/tmp/acme-artifact-ratings")
)


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
    if boto3 is None:
        raise RuntimeError("boto3 is required to store ratings")
    client = boto3.client("s3")
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(rating).encode("utf-8"),
        ContentType="application/json",
    )


def load_rating(artifact_id: str) -> Dict[str, Any] | None:
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
    if boto3 is None:
        raise RuntimeError("boto3 is required to load ratings")
    client = boto3.client("s3")
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except Exception:
        return None
    return json.loads(response["Body"].read())
