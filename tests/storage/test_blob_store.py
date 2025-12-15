"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for artifact blob stores.
"""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from src.storage import blob_store as blob_store_module
from src.storage.blob_store import (BlobNotFoundError, BlobStoreError,
                                    DownloadLink, LocalArtifactBlobStore,
                                    S3ArtifactBlobStore, StoredArtifact,
                                    build_blob_store_from_env)
from src.storage.errors import ValidationError


def test_local_blob_store_writes_file(tmp_path: Path) -> None:
    """
    test_local_blob_store_writes_file: Function description.
    :param tmp_path:
    :returns:
    """

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"binary-data")
    store = LocalArtifactBlobStore(tmp_path / "artifacts")

    result = store.store_file("artifact123", payload)
    assert isinstance(result, StoredArtifact)
    destination = tmp_path / "artifacts" / "artifact123"
    assert destination.read_bytes() == payload.read_bytes()
    assert result.bytes_written == payload.stat().st_size
    link = store.generate_download_url("artifact123")
    assert isinstance(link, DownloadLink)
    assert link.url.startswith("file://")


def test_local_blob_store_missing_download(tmp_path: Path) -> None:
    """
    test_local_blob_store_missing_download: Function description.
    :param tmp_path:
    :returns:
    """

    store = LocalArtifactBlobStore(tmp_path)
    with pytest.raises(BlobNotFoundError):
        store.generate_download_url("missing")


def test_local_blob_store_store_directory(tmp_path: Path) -> None:
    """
    test_local_blob_store_store_directory: Function description.
    :param tmp_path:
    :returns:
    """

    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "file.txt").write_text("content")
    store = LocalArtifactBlobStore(tmp_path / "artifacts")
    result = store.store_directory("artifact123", source_dir)
    assert isinstance(result, StoredArtifact)
    archive_path = tmp_path / "artifacts" / "artifact123"
    assert archive_path.exists()
    with tarfile.open(archive_path, "r:gz") as tar:
        names = tar.getnames()
    normalized = [name.replace("./", "") for name in names]
    assert "file.txt" in normalized


class _FakeS3Client:
    """
    _FakeS3Client: Class description.
    """

    def __init__(self) -> None:
        """
        __init__: Function description.
        :param:
        :returns:
        """

        self.uploads: list[tuple[str, str, Optional[Dict[str, Any]]]] = []
        self.head_calls: list[tuple[str, str]] = []
        self.presigned: dict[str, str] = {}
        self.multipart_uploads: list[dict[str, Any]] = []
        self.parts: list[dict[str, Any]] = []
        self.completed: list[dict[str, Any]] = []
        self.aborted: list[dict[str, Any]] = []
        self.puts: list[dict[str, Any]] = []

    def upload_fileobj(
        self,
        fileobj,
        bucket: str,
        key: str,
        *,
        ExtraArgs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        upload_fileobj: Function description.
        :param fileobj:
        :param bucket:
        :param key:
        :param ExtraArgs:
        :returns:
        """

        self.uploads.append((bucket, key, ExtraArgs))

    def head_object(self, *, Bucket: str, Key: str) -> None:
        """
        head_object: Function description.
        :param Bucket:
        :param Key:
        :returns:
        """

        self.head_calls.append((Bucket, Key))
        if Key.endswith("missing"):
            raise RuntimeError("not found")

    def generate_presigned_url(self, *args: Any, **kwargs: Any) -> str:
        """
        generate_presigned_url: Function description.
        :param *args:
        :param **kwargs:
        :returns:
        """

        key = kwargs.get("Params", {}).get("Key", "")
        if key.endswith("error"):
            raise RuntimeError("boom")
        return f"https://example.com/{key}"

    def create_multipart_upload(self, **kwargs: Any) -> dict[str, str]:
        """
        create_multipart_upload: Function description.
        :param **kwargs:
        :returns:
        """

        self.multipart_uploads.append(kwargs)
        return {"UploadId": "upload-1"}

    def upload_part(self, **kwargs: Any) -> dict[str, str]:
        """
        upload_part: Function description.
        :param **kwargs:
        :returns:
        """

        self.parts.append(kwargs)
        return {"ETag": f"etag-{kwargs['PartNumber']}"}

    def complete_multipart_upload(self, **kwargs: Any) -> None:
        """
        complete_multipart_upload: Function description.
        :param **kwargs:
        :returns:
        """

        self.completed.append(kwargs)

    def abort_multipart_upload(self, **kwargs: Any) -> None:
        """
        abort_multipart_upload: Function description.
        :param **kwargs:
        :returns:
        """

        self.aborted.append(kwargs)

    def put_object(self, **kwargs: Any) -> None:
        """
        put_object: Function description.
        :param **kwargs:
        :returns:
        """

        self.puts.append(kwargs)


def test_s3_blob_store_requires_bucket() -> None:
    """
    test_s3_blob_store_requires_bucket: Function description.
    :param:
    :returns:
    """

    with pytest.raises(ValidationError):
        S3ArtifactBlobStore(bucket="")


def test_s3_blob_store_store_file_and_download_url() -> None:
    """
    test_s3_blob_store_store_file_and_download_url: Function description.
    :param:
    :returns:
    """

    client = _FakeS3Client()
    store = S3ArtifactBlobStore(
        bucket="bucket",
        object_prefix="prefix",
        client=client,
    )

    payload = Path(__file__).resolve()
    stored = store.store_file("artifact123", payload, content_type="text/plain")

    assert isinstance(stored, StoredArtifact)
    assert stored.uri.startswith("s3://bucket/prefix/artifact123")
    assert client.uploads and client.uploads[0][0] == "bucket"
    assert client.uploads[0][1] == "prefix/artifact123"

    link = store.generate_download_url("artifact123", expires_in=60)
    assert isinstance(link, DownloadLink)
    assert link.url.endswith("prefix/artifact123")


def test_s3_blob_store_generate_download_url_missing_raises() -> None:
    """
    test_s3_blob_store_generate_download_url_missing_raises: Function description.
    :param:
    :returns:
    """

    client = _FakeS3Client()
    store = S3ArtifactBlobStore(bucket="bucket", client=client)

    with pytest.raises(BlobNotFoundError):
        store.generate_download_url("missing")


def test_s3_blob_store_generate_download_url_presign_errors() -> None:
    """
    test_s3_blob_store_generate_download_url_presign_errors: Function description.
    :param:
    :returns:
    """

    client = _FakeS3Client()
    store = S3ArtifactBlobStore(bucket="bucket", client=client)
    client.head_object(Bucket="bucket", Key="error")

    with pytest.raises(BlobStoreError):
        store.generate_download_url("error")


def test_s3_blob_store_store_directory_uses_multipart_upload(tmp_path: Path) -> None:
    """
    test_s3_blob_store_store_directory_uses_multipart_upload: Function description.
    :param tmp_path:
    :returns:
    """

    client = _FakeS3Client()
    store = S3ArtifactBlobStore(bucket="bucket", client=client)

    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "file.txt").write_text("content", encoding="utf-8")

    stored = store.store_directory("artifact123", source_dir)

    assert stored.bytes_written > 0
    assert client.multipart_uploads
    assert client.completed


def test_build_blob_store_from_env_selects_local_and_s3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_build_blob_store_from_env_selects_local_and_s3: Function description.
    :param tmp_path:
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("AWS_SAM_LOCAL", "1")
    monkeypatch.setenv("ARTIFACT_STORAGE_DIR", str(tmp_path))
    store = build_blob_store_from_env()
    assert isinstance(store, LocalArtifactBlobStore)

    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)
    monkeypatch.setenv("ARTIFACT_STORAGE_BUCKET", "bucket")
    monkeypatch.setenv("ARTIFACT_STORAGE_PREFIX", "prefix")
    store = build_blob_store_from_env()
    assert isinstance(store, S3ArtifactBlobStore)


def test_s3_blob_store_infers_endpoint_from_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_s3_blob_store_infers_endpoint_from_region: Function description.
    :param monkeypatch:
    :returns:
    """

    calls: Dict[str, Any] = {}

    class FakeBoto3:
        @staticmethod
        def client(service: str, **kwargs: Any) -> Any:
            """
            client: Function description.
            :param service:
            :param **kwargs:
            :returns:
            """

            calls["service"] = service
            calls["kwargs"] = kwargs
            return _FakeS3Client()

    monkeypatch.setitem(sys.modules, "boto3", FakeBoto3)
    monkeypatch.setenv("ARTIFACT_STORAGE_REGION", "us-east-1")
    monkeypatch.delenv("ARTIFACT_STORAGE_ENDPOINT", raising=False)

    store = S3ArtifactBlobStore(bucket="bucket", client=None)
    assert isinstance(store, S3ArtifactBlobStore)
    assert calls["service"] == "s3"
    assert calls["kwargs"]["region_name"] == "us-east-1"
    assert calls["kwargs"]["endpoint_url"] == "https://s3.us-east-1.amazonaws.com"


def test_s3_blob_store_respects_endpoint_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_s3_blob_store_respects_endpoint_override: Function description.
    :param monkeypatch:
    :returns:
    """

    calls: Dict[str, Any] = {}

    class FakeBoto3:
        @staticmethod
        def client(service: str, **kwargs: Any) -> Any:
            """
            client: Function description.
            :param service:
            :param **kwargs:
            :returns:
            """

            calls["kwargs"] = kwargs
            return _FakeS3Client()

    monkeypatch.setitem(sys.modules, "boto3", FakeBoto3)
    monkeypatch.setenv("ARTIFACT_STORAGE_REGION", "us-east-1")
    monkeypatch.setenv("ARTIFACT_STORAGE_ENDPOINT", "http://localhost:9000")

    S3ArtifactBlobStore(bucket="bucket", client=None)
    assert calls["kwargs"]["region_name"] == "us-east-1"
    assert calls["kwargs"]["endpoint_url"] == "http://localhost:9000"


def test_s3_blob_store_store_file_raises_on_upload_failure(
    tmp_path: Path,
) -> None:
    """
    test_s3_blob_store_store_file_raises_on_upload_failure: Function description.
    :param tmp_path:
    :returns:
    """

    class ExplodingClient(_FakeS3Client):
        """
        ExplodingClient: Class description.
        """

        def upload_fileobj(self, fileobj, bucket: str, key: str, *, ExtraArgs=None) -> None:  # type: ignore[override]
            """
            upload_fileobj: Function description.
            :param fileobj:
            :param bucket:
            :param key:
            :param ExtraArgs:
            :returns:
            """

            raise RuntimeError("boom")

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"data")
    store = S3ArtifactBlobStore(bucket="bucket", client=ExplodingClient())

    with pytest.raises(BlobStoreError, match="S3 upload failed"):
        store.store_file("artifact123", payload)


def test_s3_store_directory_aborts_on_tar_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_s3_store_directory_aborts_on_tar_failure: Function description.
    :param tmp_path:
    :param monkeypatch:
    :returns:
    """

    client = _FakeS3Client()
    store = S3ArtifactBlobStore(bucket="bucket", client=client)
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "file.txt").write_text("content", encoding="utf-8")

    def boom(*args: Any, **kwargs: Any) -> Any:
        """
        boom: Function description.
        :param *args:
        :param **kwargs:
        :returns:
        """

        raise RuntimeError("tar failure")

    monkeypatch.setattr(blob_store_module.tarfile, "open", boom)

    with pytest.raises(BlobStoreError, match="S3 upload failed"):
        store.store_directory("artifact123", source_dir)


def test_multipart_writer_puts_empty_object_and_handles_abort_exceptions() -> None:
    """
    test_multipart_writer_puts_empty_object_and_handles_abort_exceptions: Function description.
    :param:
    :returns:
    """

    class AbortExplodingClient(_FakeS3Client):
        """
        AbortExplodingClient: Class description.
        """

        def abort_multipart_upload(self, **kwargs: Any) -> None:  # type: ignore[override]
            """
            abort_multipart_upload: Function description.
            :param **kwargs:
            :returns:
            """

            raise RuntimeError("abort boom")

    client = AbortExplodingClient()
    writer = blob_store_module._S3MultipartUploadWriter(
        client,
        bucket="bucket",
        key="key",
        content_type="application/gzip",
    )
    assert writer.readable() is False
    assert writer.writable() is True
    assert writer.seekable() is False

    writer.close()
    assert client.puts and client.puts[0]["Key"] == "key"

    upload_id = writer._upload_id
    writer.abort()
    assert writer._upload_id == ""
    assert upload_id


def test_multipart_writer_raises_when_writing_after_close() -> None:
    """
    test_multipart_writer_raises_when_writing_after_close: Function description.
    :param:
    :returns:
    """

    client = _FakeS3Client()
    writer = blob_store_module._S3MultipartUploadWriter(
        client,
        bucket="bucket",
        key="key",
        content_type="application/gzip",
    )
    writer.close()

    with pytest.raises(ValueError, match="closed upload writer"):
        writer.write(b"data")


def test_multipart_writer_close_aborts_when_upload_fails() -> None:
    """
    test_multipart_writer_close_aborts_when_upload_fails: Function description.
    :param:
    :returns:
    """

    class ExplodingUploadClient(_FakeS3Client):
        """
        ExplodingUploadClient: Class description.
        """

        def upload_part(self, **kwargs: Any) -> dict[str, str]:  # type: ignore[override]
            """
            upload_part: Function description.
            :param **kwargs:
            :returns:
            """

            raise RuntimeError("upload part failed")

    client = ExplodingUploadClient()
    writer = blob_store_module._S3MultipartUploadWriter(
        client,
        bucket="bucket",
        key="key",
        content_type="application/gzip",
    )
    writer.write(b"data")

    with pytest.raises(RuntimeError, match="upload part failed"):
        writer.close()


def test_s3_blob_store_raises_when_boto3_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_s3_blob_store_raises_when_boto3_missing: Function description.
    :param monkeypatch:
    :returns:
    """

    import builtins

    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        """
        fake_import: Function description.
        :param name:
        :param *args:
        :param **kwargs:
        :returns:
        """

        if name == "boto3":
            raise ImportError("no boto3")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(BlobStoreError, match="boto3 is required for S3 artifact storage"):
        S3ArtifactBlobStore(bucket="bucket", client=None)
