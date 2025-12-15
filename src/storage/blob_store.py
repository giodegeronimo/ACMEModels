"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Artifact blob storage abstractions.
"""

from __future__ import annotations

import io
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Protocol, cast

from .errors import ValidationError


class BlobStoreError(RuntimeError):
    """Raised when binary storage fails."""


class BlobNotFoundError(BlobStoreError):
    """Raised when a requested blob cannot be located."""


class BlobStoreUnavailableError(BlobStoreError):
    """Raised when blob storage is temporarily unavailable (e.g. S3 outage)."""


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

    def store_directory(
        self,
        artifact_id: str,
        directory: Path,
        *,
        content_type: str | None = None,
    ) -> StoredArtifact:
        """Persist a directory by creating a streamed archive."""

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
        """
        __init__: Function description.
        :param base_dir:
        :returns:
        """

        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _artifact_path(self, artifact_id: str) -> Path:
        """
        _artifact_path: Function description.
        :param artifact_id:
        :returns:
        """

        return self._base_dir / artifact_id

    def store_file(
        self,
        artifact_id: str,
        file_path: Path,
        *,
        content_type: str | None = None,
    ) -> StoredArtifact:
        """
        store_file: Function description.
        :param artifact_id:
        :param file_path:
        :param content_type:
        :returns:
        """

        destination = self._artifact_path(artifact_id)
        shutil.copyfile(file_path, destination)
        bytes_written = destination.stat().st_size
        return StoredArtifact(
            artifact_id=artifact_id,
            uri=str(destination),
            bytes_written=bytes_written,
            content_type=content_type,
        )

    def store_directory(
        self,
        artifact_id: str,
        directory: Path,
        *,
        content_type: str | None = "application/gzip",
    ) -> StoredArtifact:
        """
        store_directory: Function description.
        :param artifact_id:
        :param directory:
        :param content_type:
        :returns:
        """

        temp_tar = Path(tempfile.NamedTemporaryFile(delete=False).name)
        try:
            with tarfile.open(temp_tar, "w:gz") as tar:
                tar.add(directory, arcname=".")
            return self.store_file(
                artifact_id, temp_tar, content_type=content_type
            )
        finally:
            temp_tar.unlink(missing_ok=True)

    def generate_download_url(
        self, artifact_id: str, *, expires_in: int = 900
    ) -> DownloadLink:
        """
        generate_download_url: Function description.
        :param artifact_id:
        :param expires_in:
        :returns:
        """

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
        """
        __init__: Function description.
        :param bucket:
        :param object_prefix:
        :param client:
        :returns:
        """

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
            region = (
                os.environ.get("ARTIFACT_STORAGE_REGION")
                or os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
            )
            endpoint_override = os.environ.get("ARTIFACT_STORAGE_ENDPOINT")
            client_kwargs: dict[str, Any] = {}
            if region:
                client_kwargs["region_name"] = region
                if not endpoint_override:
                    endpoint_override = f"https://s3.{region}.amazonaws.com"
            if endpoint_override:
                client_kwargs["endpoint_url"] = endpoint_override
            client = _boto3.client("s3", **client_kwargs)
        self._s3 = client

    def _object_key(self, artifact_id: str) -> str:
        """
        _object_key: Function description.
        :param artifact_id:
        :returns:
        """

        prefix = f"{self._object_prefix}/" if self._object_prefix else ""
        return f"{prefix}{artifact_id}"

    def store_file(
        self,
        artifact_id: str,
        file_path: Path,
        *,
        content_type: str | None = None,
    ) -> StoredArtifact:
        """
        store_file: Function description.
        :param artifact_id:
        :param file_path:
        :param content_type:
        :returns:
        """

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
            if _looks_like_transient_cloud_failure(exc):
                raise BlobStoreUnavailableError(
                    f"S3 temporarily unavailable: {exc}"
                ) from exc
            raise BlobStoreError(f"S3 upload failed: {exc}") from exc

        bytes_written = file_path.stat().st_size
        return StoredArtifact(
            artifact_id=artifact_id,
            uri=f"s3://{self._bucket}/{key}",
            bytes_written=bytes_written,
            content_type=content_type,
        )

    def store_directory(
        self,
        artifact_id: str,
        directory: Path,
        *,
        content_type: str | None = "application/gzip",
    ) -> StoredArtifact:
        """
        store_directory: Function description.
        :param artifact_id:
        :param directory:
        :param content_type:
        :returns:
        """

        key = self._object_key(artifact_id)
        writer = _S3MultipartUploadWriter(
            self._s3,
            bucket=self._bucket,
            key=key,
            content_type=content_type or "application/gzip",
        )
        try:
            with writer:
                fileobj = cast(IO[bytes], writer)
                with tarfile.open(fileobj=fileobj, mode="w:gz") as tar:
                    tar.add(directory, arcname=".")
        except Exception as exc:  # noqa: BLE001
            writer.abort()
            if _looks_like_transient_cloud_failure(exc):
                raise BlobStoreUnavailableError(
                    f"S3 temporarily unavailable: {exc}"
                ) from exc
            raise BlobStoreError(f"S3 upload failed: {exc}") from exc
        return StoredArtifact(
            artifact_id=artifact_id,
            uri=f"s3://{self._bucket}/{key}",
            bytes_written=writer.bytes_written,
            content_type=content_type,
        )

    def generate_download_url(
        self, artifact_id: str, *, expires_in: int = 900
    ) -> DownloadLink:
        """
        generate_download_url: Function description.
        :param artifact_id:
        :param expires_in:
        :returns:
        """

        key = self._object_key(artifact_id)
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            if _looks_like_transient_cloud_failure(exc):
                raise BlobStoreUnavailableError(
                    f"S3 temporarily unavailable: {exc}"
                ) from exc
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
            if _looks_like_transient_cloud_failure(exc):
                raise BlobStoreUnavailableError(
                    f"S3 temporarily unavailable: {exc}"
                ) from exc
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
    """
    _resolve_local_storage_dir: Function description.
    :param:
    :returns:
    """

    storage_dir_raw = (
        os.environ.get("ARTIFACT_STORAGE_DIR") or "/tmp/acme-artifacts"
    )
    storage_dir = Path(storage_dir_raw)
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


class _S3MultipartUploadWriter(io.RawIOBase):
    """File-like writer streaming directly into an S3 multipart upload."""

    def __init__(
        self,
        s3_client: Any,
        *,
        bucket: str,
        key: str,
        content_type: str,
        part_size: int = 8 * 1024 * 1024,
    ) -> None:
        """
        __init__: Function description.
        :param s3_client:
        :param bucket:
        :param key:
        :param content_type:
        :param part_size:
        :returns:
        """

        super().__init__()
        self._s3 = s3_client
        self._bucket = bucket
        self._key = key
        self._content_type = content_type
        self._part_size = max(part_size, 5 * 1024 * 1024)
        self._buffer = bytearray()
        self._parts: list[dict[str, Any]] = []
        self._part_number = 1
        self._upload_id = self._s3.create_multipart_upload(
            Bucket=self._bucket,
            Key=self._key,
            ContentType=self._content_type,
        )["UploadId"]
        self.bytes_written = 0
        self._closed = False

    def readable(self) -> bool:
        """
        readable: Function description.
        :param:
        :returns:
        """

        return False

    def writable(self) -> bool:
        """
        writable: Function description.
        :param:
        :returns:
        """

        return True

    def seekable(self) -> bool:
        """
        seekable: Function description.
        :param:
        :returns:
        """

        return False

    def write(self, data: Any) -> int:
        """
        write: Function description.
        :param data:
        :returns:
        """

        if self._closed:
            raise ValueError("Cannot write to closed upload writer")
        self._buffer.extend(memoryview(data))
        while len(self._buffer) >= self._part_size:
            chunk = memoryview(self._buffer)[: self._part_size]
            self._upload_part(bytes(chunk))
            del self._buffer[: self._part_size]
        self.bytes_written += len(data)
        return len(data)

    def _upload_part(self, data: bytes) -> None:
        """
        _upload_part: Function description.
        :param data:
        :returns:
        """

        response = self._s3.upload_part(
            Bucket=self._bucket,
            Key=self._key,
            PartNumber=self._part_number,
            UploadId=self._upload_id,
            Body=data,
        )
        self._parts.append(
            {"ETag": response["ETag"], "PartNumber": self._part_number}
        )
        self._part_number += 1

    def close(self) -> None:
        """
        close: Function description.
        :param:
        :returns:
        """

        if self._closed:
            return
        try:
            if self._buffer:
                self._upload_part(bytes(self._buffer))
                self._buffer.clear()
            if self._parts:
                self._s3.complete_multipart_upload(
                    Bucket=self._bucket,
                    Key=self._key,
                    UploadId=self._upload_id,
                    MultipartUpload={"Parts": self._parts},
                )
            else:
                self._s3.put_object(
                    Bucket=self._bucket,
                    Key=self._key,
                    Body=b"",
                    ContentType=self._content_type,
                )
        except Exception:
            self.abort()
            raise
        finally:
            self._closed = True

    def abort(self) -> None:
        """
        abort: Function description.
        :param:
        :returns:
        """

        if not self._upload_id:
            return
        try:
            self._s3.abort_multipart_upload(
                Bucket=self._bucket,
                Key=self._key,
                UploadId=self._upload_id,
            )
        except Exception:
            pass
        finally:
            self._upload_id = ""

    def __enter__(self) -> "_S3MultipartUploadWriter":
        """
        __enter__: Function description.
        :param:
        :returns:
        """

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        __exit__: Function description.
        :param exc_type:
        :param exc:
        :param tb:
        :returns:
        """

        self.close()
