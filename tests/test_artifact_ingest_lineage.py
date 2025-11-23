"""Tests to verify lineage extraction during artifact ingest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, cast

import pytest

from backend.src.handlers.artifact_create import app as handler
from src.models.artifacts import Artifact
from src.models.lineage import ArtifactLineageGraph
from src.storage.artifact_ingest import ArtifactBundle
from src.storage.blob_store import ArtifactBlobStore
from src.storage.memory import get_lineage_repo
from src.storage.metadata_store import ArtifactMetadataStore
from src.storage.name_index import NameIndexStore


@pytest.fixture(autouse=True)
def _reset_handler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Prepare handler stores and monkeypatches for tests."""
    handler._METADATA_STORE = cast(ArtifactMetadataStore, _FakeMetadataStore())
    handler._BLOB_STORE = cast(  # type: ignore[attr-defined]
        ArtifactBlobStore, _FakeBlobStore()
    )
    handler._NAME_INDEX = cast(NameIndexStore, _FakeNameIndexStore())
    monkeypatch.setenv("ACME_DISABLE_ASYNC", "1")
    monkeypatch.setenv("AWS_SAM_LOCAL", "1")


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
    ):
        self.saved_files.append((artifact_id, file_path))
        return None

    def store_directory(
        self,
        artifact_id: str,
        directory: Path,
        *,
        content_type: str | None = None,
    ):
        self.saved_dirs.append((artifact_id, directory))
        return None

    def generate_download_url(self, artifact_id: str) -> str:
        # pragma: no cover - not used
        raise NotImplementedError(
            f"download_url not implemented for {artifact_id}"
        )


class _FakeMetadataStore(ArtifactMetadataStore):
    def __init__(self) -> None:
        self.records: dict[str, Artifact] = {}

    def save(self, artifact: Artifact, *, overwrite: bool = False) -> None:
        self.records[artifact.metadata.id] = artifact

    def load(self, artifact_id: str) -> Artifact:
        return self.records[artifact_id]


class _FakeNameIndexStore:
    def __init__(self) -> None:
        self.entries: list[Any] = []

    def save(self, entry: Any) -> None:
        self.entries.append(entry)

    def delete(self, name: str) -> None:  # pragma: no cover - not used
        self.entries = [
            e for e in self.entries if getattr(e, "name", None) != name
        ]

    def scan(self) -> list[Any]:  # pragma: no cover - not used
        return list(self.entries)


def _event(body: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "pathParameters": {"artifact_type": "model"},
        "headers": {"X-Authorization": "token"},
        "body": json.dumps(body or {"url": "https://example.com/model"}),
    }


def test_sync_ingest_upserts_lineage(monkeypatch, tmp_path: Path) -> None:
    """When ingesting synchronously, lineage.json in the bundle is upserted."""
    # Ensure sync path
    monkeypatch.setattr(
        handler,
        "_can_process_synchronously",
        lambda url: True,
    )

    # Create a directory bundle with lineage.json
    bundle_dir = tmp_path / "bundle_dir"
    bundle_dir.mkdir()
    lineage = {
        "nodes": [{"artifact_id": "placeholder", "name": "n"}],
        "edges": [],
    }
    (bundle_dir / "lineage.json").write_text(
        json.dumps(lineage),
        encoding="utf-8",
    )

    def _fake_prepare(url: str) -> ArtifactBundle:
        return ArtifactBundle(
            kind="directory",
            path=bundle_dir,
            cleanup_root=bundle_dir,
        )

    monkeypatch.setattr(handler, "prepare_artifact_bundle", _fake_prepare)

    response = handler.lambda_handler(
        _event(),
        context=type("Ctx", (), {"invoked_function_arn": "arn"}),
    )
    assert response["statusCode"] in {201, 202}
    body = json.loads(response["body"])
    artifact_id = body["metadata"]["id"]

    # Verify lineage repo has graph for artifact
    graph: ArtifactLineageGraph = get_lineage_repo().get(artifact_id)
    assert graph.nodes


def test_async_ingest_upserts_lineage(monkeypatch, tmp_path: Path) -> None:
    """When ingesting via async worker, lineage is upserted."""
    # Force async path and immediate processing
    monkeypatch.setattr(
        handler,
        "_can_process_synchronously",
        lambda url: False,
    )
    monkeypatch.setenv("ACME_DISABLE_ASYNC", "1")

    # Build bundle with lineage.json
    bundle_dir = tmp_path / "bundle_async"
    bundle_dir.mkdir()
    lineage = {"nodes": [{"artifact_id": "x", "name": "n"}], "edges": []}
    (bundle_dir / "lineage.json").write_text(
        json.dumps(lineage),
        encoding="utf-8",
    )

    def _fake_prepare(url: str) -> ArtifactBundle:
        return ArtifactBundle(
            kind="directory",
            path=bundle_dir,
            cleanup_root=bundle_dir,
        )

    monkeypatch.setattr(handler, "prepare_artifact_bundle", _fake_prepare)

    response = handler.lambda_handler(
        _event(),
        context=type("Ctx", (), {"invoked_function_arn": "arn"}),
    )
    assert response["statusCode"] in {201, 202}
    body = json.loads(response["body"])
    artifact_id = body["metadata"]["id"]

    graph = get_lineage_repo().get(artifact_id)
    assert graph.nodes
