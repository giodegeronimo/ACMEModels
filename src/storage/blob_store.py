"""Artifact blob storage abstractions."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .errors import ValidationError


class BlobStoreError(RuntimeError):
    """Raised when binary storage fails."""


class BlobNotFoundError(BlobStoreError):
    """Raised when a requested blob cannot be located."""


@dataclass(frozen=True)
class StoredArtifact:
    """Metadata about a stored artifact binary."""

    artifact_id: str
    uri: str
    bytes_written: int
    content_type: str | None = None


@dataclass(frozen=True)
class DownloadLink:
    """Download link metadata returned by blob stores."""

    artifact_id: str
    url: str
    expires_in: int


class ArtifactBlobStore(Protocol):
    """Interface implemented by concrete blob stores."""

    def store_file(
        self,
        artifact_id: str,
        file_path: Path,
        *,
        content_type: str | None = None,
    ) -> StoredArtifact:
        """Persist ``file_path`` under the given artifact id."""

    def generate_download_url(
        self,
        artifact_id: str,
        *,
        expires_in: int = 900,
    ) -> DownloadLink:
        """Return a download URL (presigned for remote stores)."""


class LocalArtifactBlobStore:
    """Persist artifacts to a local directory (useful for dev/tests)."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _artifact_path(self, artifact_id: str) -> Path:
        return self._base_dir / artifact_id

    def store_file(
        self,
        artifact_id: str,
        file_path: Path,
        *,
        content_type: str | None = None,
    ) -> StoredArtifact:
        destination = self._artifact_path(artifact_id)
        shutil.copyfile(file_path, destination)
        bytes_written = destination.stat().st_size
        return StoredArtifact(
            artifact_id=artifact_id,
            uri=str(destination),
            bytes_written=bytes_written,
            content_type=content_type,
        )

    def generate_download_url(
        self, artifact_id: str, *, expires_in: int = 900
    ) -> DownloadLink:
        destination = self._artifact_path(artifact_id)
        if not destination.exists():
            raise BlobNotFoundError(
                f"Artifact '{artifact_id}' binary not found locally"
            )
        return DownloadLink(
            artifact_id=artifact_id,
            url=destination.resolve().as_uri(),
            expires_in=expires_in,
        )


class S3ArtifactBlobStore:
    """Store artifacts in an S3 bucket."""

    def __init__(
        self,
        bucket: str,
        *,
        object_prefix: str = "",
        client: Any | None = None,
    ) -> None:
        if not bucket:
            raise ValidationError("bucket name must be provided")
        self._bucket = bucket
        self._object_prefix = object_prefix.strip("/")
        if client is None:
            try:
                import boto3 as _boto3
            except ImportError as exc:  # pragma: no cover
                raise BlobStoreError(
                    "boto3 is required for S3 artifact storage"
                ) from exc
            client = _boto3.client("s3")
        self._s3 = client

    def _object_key(self, artifact_id: str) -> str:
        prefix = f"{self._object_prefix}/" if self._object_prefix else ""
        return f"{prefix}{artifact_id}"

    def store_file(
        self,
        artifact_id: str,
        file_path: Path,
        *,
        content_type: str | None = None,
    ) -> StoredArtifact:
        key = self._object_key(artifact_id)
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        try:
            with open(file_path, "rb") as handle:
                self._s3.upload_fileobj(
                    handle,
                    self._bucket,
                    key,
                    ExtraArgs=extra_args or None,
                )
        except Exception as exc:  # noqa: BLE001
            raise BlobStoreError(f"S3 upload failed: {exc}") from exc

        bytes_written = file_path.stat().st_size
        return StoredArtifact(
            artifact_id=artifact_id,
            uri=f"s3://{self._bucket}/{key}",
            bytes_written=bytes_written,
            content_type=content_type,
        )

    def generate_download_url(
        self, artifact_id: str, *, expires_in: int = 900
    ) -> DownloadLink:
        key = self._object_key(artifact_id)
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            raise BlobNotFoundError(
                f"Artifact '{artifact_id}' binary does not exist"
            ) from exc

        try:
            url = self._s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires_in,
            )
        except Exception as exc:  # noqa: BLE001
            raise BlobStoreError(
                f"Failed to generate download URL: {exc}"
            ) from exc
        return DownloadLink(
            artifact_id=artifact_id,
            url=url,
            expires_in=expires_in,
        )


def build_blob_store_from_env() -> ArtifactBlobStore:
    """Factory that mirrors the logic used in the Lambda handlers."""

    if os.environ.get("AWS_SAM_LOCAL"):
        return LocalArtifactBlobStore(_resolve_local_storage_dir())

    bucket = os.environ.get("ARTIFACT_STORAGE_BUCKET")
    if bucket:
        prefix = os.environ.get("ARTIFACT_STORAGE_PREFIX", "")
        return S3ArtifactBlobStore(
            bucket=bucket,
            object_prefix=prefix,
        )

    storage_dir = _resolve_local_storage_dir()
    return LocalArtifactBlobStore(storage_dir)


def _resolve_local_storage_dir() -> Path:
    storage_dir_raw = os.environ.get("ARTIFACT_STORAGE_DIR") \
                      or "/tmp/acme-artifacts"
    storage_dir = Path(storage_dir_raw)
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir
