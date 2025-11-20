"""Tests for the POST /artifact/{artifact_type} Lambda handler."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, cast
from uuid import uuid4

import pytest

from backend.src.handlers.artifact_create import app as handler
from src.models.artifacts import (Artifact, ArtifactData, ArtifactMetadata,
                                  ArtifactType)
from src.storage.artifact_ingest import ArtifactBundle
from src.storage.blob_store import BlobStoreError, DownloadLink, StoredArtifact
from src.storage.errors import ArtifactNotFound, ValidationError
from src.storage.metadata_store import ArtifactMetadataStore
from src.storage.name_index import NameIndexEntry


@pytest.fixture(autouse=True)
def _reset_handler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    handler._METADATA_STORE = cast(
        ArtifactMetadataStore, _FakeMetadataStore()
    )
    handler._BLOB_STORE = _FakeBlobStore()  # type: ignore[attr-defined]
    handler._LAMBDA_CLIENT = None  # type: ignore[attr-defined]
    handler._NAME_INDEX = _FakeNameIndexStore()
    monkeypatch.setenv("ACME_DISABLE_ASYNC", "1")
    monkeypatch.setenv("AWS_SAM_LOCAL", "1")
    monkeypatch.setattr(
        handler,
        "_can_process_synchronously",
        lambda url: False,
    )

    def _fake_prepare(url: str) -> ArtifactBundle:
        bundle_path = tmp_path / f"bundle_{uuid4().hex}.bin"
        bundle_path.write_text("content", encoding="utf-8")
        return ArtifactBundle(
            kind="file",
            path=bundle_path,
            cleanup_root=bundle_path,
            content_type="application/octet-stream",
        )

    monkeypatch.setattr(handler, "prepare_artifact_bundle", _fake_prepare)
    monkeypatch.setattr(
        handler,
        "compute_model_rating",
        lambda url: {
            "name": "fake",
            "category": "MODEL",
            "net_score": 1.0,
            "net_score_latency": 0.1,
            "ramp_up_time": 1.0,
            "ramp_up_time_latency": 0.1,
            "bus_factor": 1.0,
            "bus_factor_latency": 0.1,
            "performance_claims": 1.0,
            "performance_claims_latency": 0.1,
            "license": 1.0,
            "license_latency": 0.1,
            "dataset_and_code_score": 1.0,
            "dataset_and_code_score_latency": 0.1,
            "dataset_quality": 1.0,
            "dataset_quality_latency": 0.1,
            "code_quality": 1.0,
            "code_quality_latency": 0.1,
            "reproducibility": 1.0,
            "reproducibility_latency": 0.1,
            "reviewedness": 1.0,
            "reviewedness_latency": 0.1,
            "tree_score": 1.0,
            "tree_score_latency": 0.1,
            "size_score": 1.0,
            "size_score_latency": 0.1,
        },
    )

    fake_client = _FakeLambdaClient()

    class _FakeBoto3:
        def client(self, service: str):
            assert service == "lambda"
            return fake_client

    monkeypatch.setattr(handler, "boto3", _FakeBoto3())
    handler._TEST_LAMBDA_CLIENT = fake_client  # type: ignore[attr-defined]


class _FakeBlobStore:
    def __init__(self) -> None:
        self.saved_files: list[tuple[str, Path]] = []
        self.saved_dirs: list[tuple[str, Path]] = []

    def store_file(
        self,
        artifact_id: str,
        file_path: Path,
        *,
        content_type: str | None = None,
    ) -> StoredArtifact:
        self.saved_files.append((artifact_id, file_path))
        return StoredArtifact(
            artifact_id=artifact_id,
            uri=str(file_path),
            bytes_written=1,
            content_type=content_type,
        )

    def store_directory(
        self,
        artifact_id: str,
        directory: Path,
        *,
        content_type: str | None = None,
    ) -> StoredArtifact:
        self.saved_dirs.append((artifact_id, directory))
        return StoredArtifact(
            artifact_id=artifact_id,
            uri=str(directory),
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


class _FakeMetadataStore(ArtifactMetadataStore):
    def __init__(self) -> None:
        self.records: dict[str, Artifact] = {}

    def save(self, artifact: Artifact, *, overwrite: bool = False) -> None:
        artifact_id = artifact.metadata.id
        if not overwrite and artifact_id in self.records:
            raise ValidationError(f"Artifact '{artifact_id}' already exists")
        self.records[artifact_id] = artifact

    def load(self, artifact_id: str) -> Artifact:
        try:
            return self.records[artifact_id]
        except KeyError as exc:
            raise ArtifactNotFound(
                f"Artifact '{artifact_id}' does not exist"
            ) from exc


class _FakeLambdaClient:
    def __init__(self) -> None:
        self.invocations: list[dict[str, Any]] = []

    def invoke(
        self,
        *,
        FunctionName: str,
        InvocationType: str,
        Payload: bytes,
    ) -> dict[str, Any]:
        self.invocations.append(
            {
                "function": FunctionName,
                "invocation_type": InvocationType,
                "payload": json.loads(Payload.decode("utf-8")),
            }
        )
        return {"StatusCode": 202}


class _FailingStore(_FakeBlobStore):
    def store_file(
        self,
        artifact_id: str,
        file_path: Path,
        *,
        content_type: str | None = None,
    ) -> StoredArtifact:
        raise BlobStoreError("boom")

    def store_directory(
        self,
        artifact_id: str,
        directory: Path,
        *,
        content_type: str | None = None,
    ) -> StoredArtifact:
        raise BlobStoreError("boom")


class _FakeNameIndexStore:
    def __init__(self) -> None:
        self.entries: list[NameIndexEntry] = []

    def save(self, entry: NameIndexEntry) -> None:
        self.entries.append(entry)

    def delete(self, entry: NameIndexEntry) -> None:
        self.entries = [
            existing for existing in self.entries if existing != entry
        ]

    def scan(
        self,
        *,
        start_key: Any | None = None,
        limit: int | None = None,
    ) -> tuple[list[NameIndexEntry], Any | None]:
        return list(self.entries), None


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
    context = type("Ctx", (), {"invoked_function_arn": "arn:aws:lambda:test"})
    response = handler.lambda_handler(_event(), context=context)

    assert response["statusCode"] == 201
    body = json.loads(response["body"])
    assert body["metadata"]["type"] == "model"
    fake_client = handler._TEST_LAMBDA_CLIENT  # type: ignore[attr-defined]
    assert len(fake_client.invocations) in {0, 1}


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


def test_create_artifact_accepts_name_field() -> None:
    body = {
        "url": "https://huggingface.co/org/model",
        "name": "custom-name",
    }
    context = type("Ctx", (), {"invoked_function_arn": "arn"})
    response = handler.lambda_handler(_event(body=body), context=context)

    assert response["statusCode"] in {201, 202}
    assert json.loads(response["body"])["metadata"]["name"] == "custom-name"


def test_create_artifact_ignores_unknown_fields() -> None:
    body = {
        "url": "https://huggingface.co/org/model",
        "unknown": "value",
    }
    context = type("Ctx", (), {"invoked_function_arn": "arn"})
    response = handler.lambda_handler(_event(body=body), context=context)

    assert response["statusCode"] in {201, 202}


def test_create_artifact_handles_duplicate_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate_id = "deadbeefdeadbeefdeadbeefdeadbeef"
    monkeypatch.setattr(handler, "_generate_artifact_id", lambda: duplicate_id)

    context = type("Ctx", (), {"invoked_function_arn": "arn"})

    first = handler.lambda_handler(_event(), context=context)
    assert first["statusCode"] in {201, 202}

    second = handler.lambda_handler(_event(), context=context)
    assert second["statusCode"] == 409
    assert "already exists" in json.loads(second["body"])["error"]


def test_store_artifact_records_readme_excerpt(tmp_path: Path) -> None:
    metadata = ArtifactMetadata(
        name="example",
        id="artifact-id",
        type=ArtifactType.MODEL,
    )
    artifact = Artifact(
        metadata=metadata,
        data=ArtifactData(url="https://x"),
    )

    handler._store_artifact(artifact, readme_excerpt="Readme text")

    entries = handler._NAME_INDEX.entries  # type: ignore[attr-defined]
    assert entries
    assert entries[0].readme_excerpt == "Readme text"


def test_derive_artifact_name_hf_resolve_model() -> None:
    url = (
        "https://huggingface.co/openai/whisper-tiny/resolve/main/config.json"
    )
    assert handler._derive_artifact_name(url) == "whisper-tiny"


def test_derive_artifact_name_hf_resolve_dataset() -> None:
    url = (
        "https://huggingface.co/datasets/acme/sentiment/resolve/main/data.txt"
    )
    assert handler._derive_artifact_name(url) == "sentiment"


def test_create_artifact_accepts_base64_body() -> None:
    context = type("Ctx", (), {"invoked_function_arn": "arn"})
    response = handler.lambda_handler(
        _event(is_base64=True), context=context
    )

    assert response["statusCode"] in {201, 202}


def test_create_artifact_blob_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*args, **kwargs):
        raise BlobStoreError("boom")

    monkeypatch.setattr(handler, "_enqueue_async_ingest", _fail)
    context = type("Ctx", (), {"invoked_function_arn": "arn"})
    response = handler.lambda_handler(_event(), context=context)
    assert response["statusCode"] == 502
    assert "boom" in json.loads(response["body"])["error"]


def test_create_artifact_returns_202_when_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACME_DISABLE_ASYNC", "0")
    monkeypatch.setattr(
        handler,
        "_wait_for_download_ready",
        lambda *args, **kwargs: False,
    )
    context = type("Ctx", (), {"invoked_function_arn": "arn"})
    response = handler.lambda_handler(_event(), context=context)
    assert response["statusCode"] == 202


def test_async_worker_processes_ingest() -> None:
    event = {
        "task": "ingest",
        "artifact": {
            "metadata": {
                "name": "model-name",
                "id": "artifact123",
                "type": "model",
            }
        },
        "source_url": "https://huggingface.co/org/model",
    }
    response = handler.lambda_handler(event, context={})
    assert response["statusCode"] == 200
    fake_store = handler._BLOB_STORE  # type: ignore[attr-defined]
    assert fake_store.saved_files  # type: ignore[attr-defined]
