"""Tests for the POST /artifact/{artifact_type} Lambda handler."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from backend.src.handlers.artifact_create import app as handler
from src.storage.blob_store import BlobStoreError, DownloadLink, StoredArtifact
from src.storage.memory import InMemoryArtifactRepository


@pytest.fixture(autouse=True)
def _reset_handler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    handler._REPO = InMemoryArtifactRepository()  # type: ignore[attr-defined]
    handler._BLOB_STORE = _FakeBlobStore()  # type: ignore[attr-defined]
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"x")

    def _fake_prepare(url: str):
        return bundle, "application/zip"

    monkeypatch.setattr(
        handler, "prepare_artifact_bundle", _fake_prepare
    )


class _FakeBlobStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, Path]] = []

    def store_file(
        self,
        artifact_id: str,
        file_path: Path,
        *,
        content_type: str | None = None,
    ) -> StoredArtifact:
        self.saved.append((artifact_id, file_path))
        return StoredArtifact(
            artifact_id=artifact_id,
            uri=str(file_path),
            bytes_written=1,
            content_type=content_type,
        )

    def generate_download_url(
        self, artifact_id: str, *, expires_in: int = 900
    ) -> DownloadLink:
        return DownloadLink(
            artifact_id=artifact_id,
            url=f"https://downloads/{artifact_id}?ttl={expires_in}",
            expires_in=expires_in,
        )


def _event(
    *,
    artifact_type: str = "model",
    body: Dict[str, Any] | None = None,
    is_base64: bool = False,
) -> Dict[str, Any]:
    payload = json.dumps(body or {"url": "https://huggingface.co/org/model"})
    if is_base64:
        payload = base64.b64encode(payload.encode("utf-8")).decode("utf-8")
    return {
        "pathParameters": {"artifact_type": artifact_type},
        "headers": {"X-Authorization": "placeholder"},
        "body": payload,
        "isBase64Encoded": is_base64,
    }


def test_create_artifact_success() -> None:
    response = handler.lambda_handler(_event(), context={})

    assert response["statusCode"] == 201
    body = json.loads(response["body"])
    assert body["metadata"]["type"] == "model"
    assert body["data"]["download_url"].endswith(
        body["metadata"]["id"]
    )


def test_create_artifact_rejects_invalid_type() -> None:
    response = handler.lambda_handler(
        _event(artifact_type="invalid"), context={}
    )

    assert response["statusCode"] == 400
    assert "invalid" in json.loads(response["body"])["error"]


def test_create_artifact_requires_url_field() -> None:
    bad_event = _event(body={"not_url": "value"})
    response = handler.lambda_handler(bad_event, context={})

    assert response["statusCode"] == 400
    assert "url" in json.loads(response["body"])["error"]


def test_create_artifact_handles_duplicate_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate_id = "deadbeefdeadbeefdeadbeefdeadbeef"
    monkeypatch.setattr(handler, "_generate_artifact_id", lambda: duplicate_id)

    first = handler.lambda_handler(_event(), context={})
    assert first["statusCode"] == 201

    second = handler.lambda_handler(_event(), context={})
    assert second["statusCode"] == 409
    assert "already exists" in json.loads(second["body"])["error"]


def test_create_artifact_accepts_base64_body() -> None:
    response = handler.lambda_handler(
        _event(is_base64=True), context={}
    )

    assert response["statusCode"] == 201


def test_create_artifact_blob_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingStore(_FakeBlobStore):
        def store_file(
            self,
            artifact_id: str,
            file_path: Path,
            *,
            content_type: str | None = None,
        ) -> StoredArtifact:
            raise BlobStoreError("boom")

    handler._BLOB_STORE = _FailingStore()  # type: ignore[attr-defined]
    response = handler.lambda_handler(_event(), context={})
    assert response["statusCode"] == 502
    assert "boom" in json.loads(response["body"])["error"]
