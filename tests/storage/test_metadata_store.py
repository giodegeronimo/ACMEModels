"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Unit tests for artifact metadata stores.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from src.models.artifacts import (Artifact, ArtifactData, ArtifactMetadata,
                                  ArtifactType)
from src.storage.errors import ArtifactNotFound, ValidationError
from src.storage.metadata_store import (ArtifactMetadataStore,
                                        LocalArtifactMetadataStore,
                                        MetadataStoreError,
                                        S3ArtifactMetadataStore,
                                        build_metadata_store_from_env)


def _artifact(artifact_id: str = "abc123") -> Artifact:
    """
    _artifact: Function description.
    :param artifact_id:
    :returns:
    """

    return Artifact(
        metadata=ArtifactMetadata(
            name="demo",
            id=artifact_id,
            type=ArtifactType.MODEL,
        ),
        data=ArtifactData(url="https://example.com/model"),
    )


def test_local_metadata_store_round_trip(tmp_path: Path) -> None:
    """
    test_local_metadata_store_round_trip: Function description.
    :param tmp_path:
    :returns:
    """

    store = LocalArtifactMetadataStore(tmp_path)
    artifact = _artifact()

    store.save(artifact)
    loaded = store.load(artifact.metadata.id)

    assert loaded == artifact


def test_local_metadata_store_duplicate_without_overwrite_raises(
    tmp_path: Path,
) -> None:
    """
    test_local_metadata_store_duplicate_without_overwrite_raises: Function description.
    :param tmp_path:
    :returns:
    """

    store = LocalArtifactMetadataStore(tmp_path)
    artifact = _artifact()
    store.save(artifact)

    with pytest.raises(ValidationError):
        store.save(artifact)


def test_local_metadata_store_missing_raises(tmp_path: Path) -> None:
    """
    test_local_metadata_store_missing_raises: Function description.
    :param tmp_path:
    :returns:
    """

    store = LocalArtifactMetadataStore(tmp_path)
    with pytest.raises(ArtifactNotFound):
        store.load("missing")


class _FakeClientError(Exception):
    """
    _FakeClientError: Class description.
    """

    def __init__(self, code: str) -> None:
        """
        __init__: Function description.
        :param code:
        :returns:
        """

        super().__init__(code)
        self.response = {"Error": {"Code": code}}


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

        self.objects: dict[tuple[str, str], bytes] = {}
        self.raise_on_head: Optional[Exception] = None
        self.raise_on_get: Optional[Exception] = None
        self.head_calls: list[tuple[str, str]] = []
        self.get_calls: list[tuple[str, str]] = []
        self.put_calls: list[tuple[str, str]] = []

    def head_object(self, *, Bucket: str, Key: str) -> Dict[str, Any]:
        """
        head_object: Function description.
        :param Bucket:
        :param Key:
        :returns:
        """

        self.head_calls.append((Bucket, Key))
        if self.raise_on_head is not None:
            raise self.raise_on_head
        if (Bucket, Key) not in self.objects:
            raise _FakeClientError("NoSuchKey")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def put_object(
        self, *, Bucket: str, Key: str, Body: bytes, ContentType: str
    ) -> None:
        """
        put_object: Function description.
        :param Bucket:
        :param Key:
        :param Body:
        :param ContentType:
        :returns:
        """

        self.put_calls.append((Bucket, Key))
        self.objects[(Bucket, Key)] = Body

    def get_object(self, *, Bucket: str, Key: str) -> Dict[str, Any]:
        """
        get_object: Function description.
        :param Bucket:
        :param Key:
        :returns:
        """

        self.get_calls.append((Bucket, Key))
        if self.raise_on_get is not None:
            raise self.raise_on_get
        if (Bucket, Key) not in self.objects:
            raise _FakeClientError("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


def test_s3_metadata_store_save_and_duplicate_detection() -> None:
    """
    test_s3_metadata_store_save_and_duplicate_detection: Function description.
    :param:
    :returns:
    """

    client = _FakeS3Client()
    store = S3ArtifactMetadataStore(
        bucket="bucket",
        prefix="metadata",
        client=client,
    )
    artifact = _artifact()

    store.save(artifact)
    with pytest.raises(ValidationError):
        store.save(artifact)
    store.save(artifact, overwrite=True)

    key = "metadata/abc123.json"
    assert client.put_calls
    payload = json.loads(client.objects[("bucket", key)])
    assert payload["metadata"]["id"] == "abc123"


def test_s3_metadata_store_load_not_found_maps_to_artifact_not_found() -> None:
    """
    test_s3_metadata_store_load_not_found_maps_to_artifact_not_found: Function description.
    :param:
    :returns:
    """

    client = _FakeS3Client()
    client.raise_on_get = _FakeClientError("NoSuchKey")
    store = S3ArtifactMetadataStore(bucket="bucket", prefix="metadata", client=client)

    with pytest.raises(ArtifactNotFound):
        store.load("missing")


def test_s3_metadata_store_load_access_denied_maps_to_artifact_not_found() -> None:
    """
    test_s3_metadata_store_load_access_denied_maps_to_artifact_not_found: Function description.
    :param:
    :returns:
    """

    client = _FakeS3Client()
    client.raise_on_get = _FakeClientError("AccessDenied")
    store = S3ArtifactMetadataStore(bucket="bucket", prefix="metadata", client=client)

    with pytest.raises(ArtifactNotFound):
        store.load("missing")


def test_s3_metadata_store_head_unexpected_error_raises_store_error() -> None:
    """
    test_s3_metadata_store_head_unexpected_error_raises_store_error: Function description.
    :param:
    :returns:
    """

    client = _FakeS3Client()
    client.raise_on_head = _FakeClientError("Boom")
    store = S3ArtifactMetadataStore(bucket="bucket", prefix="metadata", client=client)

    with pytest.raises(MetadataStoreError):
        store.save(_artifact())


def test_s3_metadata_store_load_success() -> None:
    """
    test_s3_metadata_store_load_success: Function description.
    :param:
    :returns:
    """

    client = _FakeS3Client()
    store = S3ArtifactMetadataStore(bucket="bucket", prefix="metadata", client=client)
    artifact = _artifact()
    store.save(artifact, overwrite=True)

    loaded = store.load("abc123")
    assert loaded == artifact


def test_s3_metadata_store_raises_when_boto3_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_s3_metadata_store_raises_when_boto3_missing: Function description.
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

    with pytest.raises(MetadataStoreError, match="boto3 is required for S3 metadata store"):
        S3ArtifactMetadataStore(bucket="bucket", client=None)


def test_build_metadata_store_from_env_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_build_metadata_store_from_env_local: Function description.
    :param tmp_path:
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("AWS_SAM_LOCAL", "1")
    monkeypatch.setenv("ARTIFACT_METADATA_DIR", str(tmp_path))

    store = build_metadata_store_from_env()

    assert isinstance(store, LocalArtifactMetadataStore)


def test_build_metadata_store_from_env_s3(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_build_metadata_store_from_env_s3: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)
    monkeypatch.setenv("ARTIFACT_STORAGE_BUCKET", "bucket")
    monkeypatch.setenv("ARTIFACT_METADATA_PREFIX", "custom-prefix")

    class _DummyStore:
        """
        _DummyStore: Class description.
        """

        def __init__(self, *, bucket: str, prefix: str, client: Any | None = None) -> None:
            """
            __init__: Function description.
            :param bucket:
            :param prefix:
            :param client:
            :returns:
            """

            self.bucket = bucket
            self.prefix = prefix

    monkeypatch.setattr(
        "src.storage.metadata_store.S3ArtifactMetadataStore",
        _DummyStore,
    )

    store = build_metadata_store_from_env()

    assert getattr(store, "bucket") == "bucket"
    assert getattr(store, "prefix") == "custom-prefix"


def test_metadata_store_abstract_methods_raise() -> None:
    """
    test_metadata_store_abstract_methods_raise: Function description.
    :param:
    :returns:
    """

    base = ArtifactMetadataStore()
    with pytest.raises(NotImplementedError):
        base.save(_artifact())
    with pytest.raises(NotImplementedError):
        base.load("abc123")


def test_s3_metadata_store_requires_bucket() -> None:
    """
    test_s3_metadata_store_requires_bucket: Function description.
    :param:
    :returns:
    """

    with pytest.raises(ValidationError):
        S3ArtifactMetadataStore(bucket="")


def test_s3_metadata_store_builds_default_client_with_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_s3_metadata_store_builds_default_client_with_region: Function description.
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

    store = S3ArtifactMetadataStore(bucket="bucket", client=None)
    assert isinstance(store, S3ArtifactMetadataStore)
    assert calls["service"] == "s3"
    assert calls["kwargs"]["region_name"] == "us-east-1"


def test_s3_metadata_store_save_wraps_put_errors() -> None:
    """
    test_s3_metadata_store_save_wraps_put_errors: Function description.
    :param:
    :returns:
    """

    class ExplodingClient(_FakeS3Client):
        """
        ExplodingClient: Class description.
        """

        def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:  # type: ignore[override]
            """
            put_object: Function description.
            :param Bucket:
            :param Key:
            :param Body:
            :param ContentType:
            :returns:
            """

            raise RuntimeError("boom")

    store = S3ArtifactMetadataStore(bucket="bucket", client=ExplodingClient())
    with pytest.raises(MetadataStoreError, match="Failed to write metadata"):
        store.save(_artifact(), overwrite=True)


def test_s3_metadata_store_load_wraps_unexpected_errors() -> None:
    """
    test_s3_metadata_store_load_wraps_unexpected_errors: Function description.
    :param:
    :returns:
    """

    client = _FakeS3Client()
    client.raise_on_get = _FakeClientError("Boom")
    store = S3ArtifactMetadataStore(bucket="bucket", client=client)
    with pytest.raises(MetadataStoreError, match="Failed to read metadata"):
        store.load("missing")


def test_metadata_store_error_helpers_handle_weird_exceptions() -> None:
    """
    test_metadata_store_error_helpers_handle_weird_exceptions: Function description.
    :param:
    :returns:
    """

    from src.storage import metadata_store as metadata_store_module

    assert metadata_store_module._is_access_denied(Exception("x")) is False
    assert metadata_store_module._is_not_found(Exception("x")) is False

    class Weird(Exception):
        """
        Weird: Class description.
        """

        def __init__(self) -> None:
            """
            __init__: Function description.
            :param:
            :returns:
            """

            self.response = "not-a-dict"

    assert metadata_store_module._is_access_denied(Weird()) is False
    assert metadata_store_module._is_not_found(Weird()) is False

    class Weird2(Exception):
        """
        Weird2: Class description.
        """

        def __init__(self) -> None:
            """
            __init__: Function description.
            :param:
            :returns:
            """

            self.response = {"Error": "nope"}

    assert metadata_store_module._is_access_denied(Weird2()) is False
    assert metadata_store_module._is_not_found(Weird2()) is False

    class Weird3(Exception):
        """
        Weird3: Class description.
        """

        def __init__(self) -> None:
            """
            __init__: Function description.
            :param:
            :returns:
            """

            self.response = {"Error": {"Code": 123}}

    assert metadata_store_module._is_access_denied(Weird3()) is False
    assert metadata_store_module._is_not_found(Weird3()) is False
