"""Tests for DELETE /reset handler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from backend.src.handlers.reset import app as handler


@pytest.fixture(autouse=True)
def _reset_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)
    monkeypatch.setenv("ARTIFACT_STORAGE_BUCKET", "test-bucket")
    monkeypatch.setenv("ARTIFACT_STORAGE_PREFIX", "artifacts")
    monkeypatch.setenv("ARTIFACT_METADATA_PREFIX", "metadata")
    monkeypatch.setattr(handler, "boto3", _FakeBoto3(tmp_path))


def _event(headers: Dict[str, str] | None = None) -> Dict[str, Any]:
    return {"headers": headers or {}}


def test_reset_calls_s3_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = handler.lambda_handler(
        _event({"X-Authorization": "token"}), {}
    )

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["status"] == "reset"


def test_reset_local_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AWS_SAM_LOCAL", "1")
    metadata_dir = tmp_path / "meta"
    artifacts_dir = tmp_path / "blobs"
    metadata_dir.mkdir()
    (metadata_dir / "foo.json").write_text("{}", encoding="utf-8")
    artifacts_dir.mkdir()
    (artifacts_dir / "blob.bin").write_bytes(b"data")
    monkeypatch.setenv("ARTIFACT_METADATA_DIR", str(metadata_dir))
    monkeypatch.setenv("ARTIFACT_STORAGE_DIR", str(artifacts_dir))

    handler.lambda_handler(_event(), {})

    assert not any(metadata_dir.iterdir())
    assert not any(artifacts_dir.iterdir())


class _FakeBoto3:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    def client(self, service: str, **kwargs):
        assert service == "s3"
        return _FakeS3Client()


class _FakeS3Client:
    def __init__(self) -> None:
        self.deleted_batches: list[list[dict[str, str]]] = []

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"

        class _Paginator:
            def paginate(self, **kwargs):
                prefix = kwargs.get("Prefix", "")
                if prefix == "artifacts":
                    return [{"Contents": [{"Key": "artifacts/a"}]}]
                if prefix == "metadata":
                    return [{"Contents": [{"Key": "metadata/a"}]}]
                return []

        return _Paginator()

    def delete_objects(self, Bucket: str, Delete: Dict[str, Any]):
        self.deleted_batches.append(Delete["Objects"])
