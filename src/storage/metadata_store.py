"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Persistent metadata storage for artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.models.artifacts import (Artifact, ArtifactData, ArtifactMetadata,
                                  ArtifactType)

from .errors import ArtifactNotFound, ValidationError

DEFAULT_METADATA_DIR = "/tmp/acme-artifact-metadata"


class MetadataStoreError(RuntimeError):
    """Raised when metadata persistence fails."""


class MetadataStoreUnavailableError(MetadataStoreError):
    """Raised when metadata storage is temporarily unavailable (e.g. S3 outage)."""


def _looks_like_transient_cloud_failure(exc: Exception) -> bool:
    """
    _looks_like_transient_cloud_failure: Function description.
    :param exc:
    :returns:
    """

    code = None
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        if isinstance(error, dict):
            code = error.get("Code")
    if isinstance(code, str) and code:
        if code in {
            "SlowDown",
            "Throttling",
            "ThrottlingException",
            "RequestTimeout",
            "RequestTimeoutException",
            "ServiceUnavailable",
            "InternalError",
            "503",
        }:
            return True
    name = exc.__class__.__name__
    if name in {
        "EndpointConnectionError",
        "ConnectTimeoutError",
        "ReadTimeoutError",
        "ConnectionClosedError",
    }:
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "timed out",
            "timeout",
            "temporarily unavailable",
            "service unavailable",
            "connection reset",
            "connection aborted",
            "connection refused",
            "endpoint connection error",
        )
    )


class ArtifactMetadataStore:
    """Interface for persisting artifact metadata records."""

    def save(self, artifact: Artifact, *, overwrite: bool = False) -> None:
        """
        save: Function description.
        :param artifact:
        :param overwrite:
        :returns:
        """

        raise NotImplementedError

    def load(self, artifact_id: str) -> Artifact:
        """
        load: Function description.
        :param artifact_id:
        :returns:
        """

        raise NotImplementedError


class LocalArtifactMetadataStore(ArtifactMetadataStore):
    """File-based metadata store (useful for local tests)."""

    def __init__(self, base_dir: Path) -> None:
        """
        __init__: Function description.
        :param base_dir:
        :returns:
        """

        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, artifact_id: str) -> Path:
        """
        _path: Function description.
        :param artifact_id:
        :returns:
        """

        return self._base_dir / f"{artifact_id}.json"

    def save(self, artifact: Artifact, *, overwrite: bool = False) -> None:
        """
        save: Function description.
        :param artifact:
        :param overwrite:
        :returns:
        """

        path = self._path(artifact.metadata.id)
        if path.exists() and not overwrite:
            raise ValidationError(
                f"Artifact '{artifact.metadata.id}' already exists"
            )
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_artifact_to_payload(artifact), handle)

    def load(self, artifact_id: str) -> Artifact:
        """
        load: Function description.
        :param artifact_id:
        :returns:
        """

        path = self._path(artifact_id)
        if not path.exists():
            raise ArtifactNotFound(f"Artifact '{artifact_id}' does not exist")
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return _payload_to_artifact(payload)


class S3ArtifactMetadataStore(ArtifactMetadataStore):
    """Persist metadata records in S3."""

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "metadata",
        client: Any | None = None,
    ) -> None:
        """
        __init__: Function description.
        :param bucket:
        :param prefix:
        :param client:
        :returns:
        """

        if not bucket:
            raise ValidationError("bucket name must be provided")
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        if client is None:
            try:
                import boto3 as _boto3
            except ImportError as exc:  # pragma: no cover
                raise MetadataStoreError(
                    "boto3 is required for S3 metadata store"
                ) from exc
            region = (
                os.environ.get("ARTIFACT_STORAGE_REGION")
                or os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
            )
            client_kwargs: dict[str, Any] = {}
            if region:
                client_kwargs["region_name"] = region
            client = _boto3.client("s3", **client_kwargs)
        self._s3 = client

    def _key(self, artifact_id: str) -> str:
        """
        _key: Function description.
        :param artifact_id:
        :returns:
        """

        prefix = f"{self._prefix}/" if self._prefix else ""
        return f"{prefix}{artifact_id}.json"

    def save(self, artifact: Artifact, *, overwrite: bool = False) -> None:
        """
        save: Function description.
        :param artifact:
        :param overwrite:
        :returns:
        """

        key = self._key(artifact.metadata.id)
        if not overwrite and self._exists(key):
            raise ValidationError(
                f"Artifact '{artifact.metadata.id}' already exists"
            )
        try:
            payload = json.dumps(_artifact_to_payload(artifact)).encode(
                "utf-8"
            )
            self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=payload,
                ContentType="application/json",
            )
        except Exception as exc:  # noqa: BLE001
            if _looks_like_transient_cloud_failure(exc):
                raise MetadataStoreUnavailableError(
                    f"S3 temporarily unavailable: {exc}"
                ) from exc
            raise MetadataStoreError(
                f"Failed to write metadata: {exc}"
            ) from exc

    def load(self, artifact_id: str) -> Artifact:
        """
        load: Function description.
        :param artifact_id:
        :returns:
        """

        key = self._key(artifact_id)
        try:
            response = self._s3.get_object(Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            if _is_access_denied(exc) or _is_not_found(exc):
                raise ArtifactNotFound(
                    f"Artifact '{artifact_id}' does not exist"
                ) from exc
            if _looks_like_transient_cloud_failure(exc):
                raise MetadataStoreUnavailableError(
                    f"S3 temporarily unavailable: {exc}"
                ) from exc
            raise MetadataStoreError(
                f"Failed to read metadata: {exc}"
            ) from exc
        payload = json.loads(response["Body"].read())
        return _payload_to_artifact(payload)

    def _exists(self, key: str) -> bool:
        """
        _exists: Function description.
        :param key:
        :returns:
        """

        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception as exc:  # noqa: BLE001
            if _is_not_found(exc):
                return False
            if _looks_like_transient_cloud_failure(exc):
                raise MetadataStoreUnavailableError(
                    f"S3 temporarily unavailable: {exc}"
                ) from exc
            raise MetadataStoreError(
                f"Failed to probe metadata: {exc}"
            ) from exc


def _is_access_denied(exc: Exception) -> bool:
    """
    _is_access_denied: Function description.
    :param exc:
    :returns:
    """

    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    if not isinstance(error, dict):
        return False
    code = error.get("Code")
    if not isinstance(code, str):
        return False
    return code == "AccessDenied"


def _is_not_found(exc: Exception) -> bool:
    """
    _is_not_found: Function description.
    :param exc:
    :returns:
    """

    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    if not isinstance(error, dict):
        return False
    code = error.get("Code")
    if not isinstance(code, str):
        return False
    return code in {"404", "NoSuchKey", "NotFound"}


def _artifact_to_payload(artifact: Artifact) -> dict[str, Any]:
    """
    _artifact_to_payload: Function description.
    :param artifact:
    :returns:
    """

    return {
        "metadata": {
            "name": artifact.metadata.name,
            "id": artifact.metadata.id,
            "type": artifact.metadata.type.value,
        },
        "data": {"url": artifact.data.url},
    }


def _payload_to_artifact(payload: dict[str, Any]) -> Artifact:
    """
    _payload_to_artifact: Function description.
    :param payload:
    :returns:
    """

    metadata_raw = payload.get("metadata") or {}
    data_raw = payload.get("data") or {}
    metadata = ArtifactMetadata(
        name=metadata_raw["name"],
        id=metadata_raw["id"],
        type=ArtifactType(metadata_raw["type"]),
    )
    data = ArtifactData(url=data_raw["url"])
    return Artifact(metadata=metadata, data=data)


def build_metadata_store_from_env() -> ArtifactMetadataStore:
    """
    build_metadata_store_from_env: Function description.
    :param:
    :returns:
    """

    if os.environ.get("AWS_SAM_LOCAL"):
        base_dir = Path(
            os.environ.get("ARTIFACT_METADATA_DIR", DEFAULT_METADATA_DIR)
        )
        return LocalArtifactMetadataStore(base_dir)

    bucket = os.environ.get("ARTIFACT_STORAGE_BUCKET")
    if bucket:
        prefix = os.environ.get("ARTIFACT_METADATA_PREFIX", "metadata")
        return S3ArtifactMetadataStore(bucket=bucket, prefix=prefix)

    base_dir = Path(
        os.environ.get("ARTIFACT_METADATA_DIR", DEFAULT_METADATA_DIR)
    )
    return LocalArtifactMetadataStore(base_dir)
