"""Tests for GET /artifact/model/{id}/lineage."""

from __future__ import annotations

import json
from typing import Any, Dict, cast

import pytest

from backend.src.handlers.artifact_lineage import app as handler
from src.models.artifacts import (Artifact, ArtifactData, ArtifactMetadata,
                                  ArtifactType)
from src.models.lineage import (ArtifactLineageEdge, ArtifactLineageGraph,
                                ArtifactLineageNode)
from src.storage.errors import ArtifactNotFound, LineageNotFound
from src.storage.lineage_store import LineageStore
from src.storage.metadata_store import ArtifactMetadataStore


@pytest.fixture(autouse=True)
def _reset_store() -> None:
    """Ensure handler metadata store is reset between tests."""
    handler._METADATA_STORE = cast(
        ArtifactMetadataStore, _FakeMetadataStore()
    )
    handler._LINEAGE_STORE = cast(LineageStore, _FakeLineageStore())


def _event(*, artifact_id: str = "abc123") -> Dict[str, Any]:
    return {
        "pathParameters": {"artifact_type": "model", "id": artifact_id},
        "headers": {"X-Authorization": "token"},
    }


def _store_artifact(*, artifact_id: str = "abc123") -> None:
    handler._METADATA_STORE.save(  # type: ignore[attr-defined]
        Artifact(
            metadata=ArtifactMetadata(
                name="demo",
                id=artifact_id,
                type=ArtifactType.MODEL,
            ),
            data=ArtifactData(url="https://example.com/model"),
        )
    )


def test_lineage_success(monkeypatch) -> None:
    """Handler returns 200 and serialized graph when lineage exists."""
    _store_artifact()

    # Prepare a simple graph and a fake lineage repo
    nodes = [
        ArtifactLineageNode(
            artifact_id="abc123",
            name="root",
        )
    ]
    edges: list[ArtifactLineageEdge] = []
    graph = ArtifactLineageGraph(nodes=nodes, edges=edges)
    handler._LINEAGE_STORE.save("abc123", graph)  # type: ignore[attr-defined]

    response = handler.lambda_handler(_event(), context={})
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert "nodes" in body and isinstance(body["nodes"], list)
    assert body["nodes"][0]["artifact_id"] == "abc123"


def test_lineage_artifact_missing() -> None:
    """Missing artifact should produce 404."""
    response = handler.lambda_handler(_event(), context={})
    assert response["statusCode"] == 404


def test_lineage_invalid_id() -> None:
    """Invalid artifact id (malformed) yields 400."""
    event = _event(artifact_id="not valid!")
    response = handler.lambda_handler(event, context={})
    assert response["statusCode"] == 400


def test_lineage_not_found_in_repo(monkeypatch) -> None:
    """When lineage repo can't find graph, handler returns 400."""
    _store_artifact()

    handler._LINEAGE_STORE = cast(
        LineageStore, _FakeLineageStore(raises=True)
    )
    response = handler.lambda_handler(_event(), context={})
    assert response["statusCode"] == 400


def test_lineage_missing_auth_enforced(monkeypatch) -> None:
    """If auth is enforced (AUTH_OPTIONAL=0) and header missing, return 403."""
    # Require auth for this test
    monkeypatch.setenv("AUTH_OPTIONAL", "0")

    _store_artifact()

    # Provide no headers
    event = {
        "pathParameters": {"artifact_type": "model", "id": "abc123"}
    }

    response = handler.lambda_handler(event, context={})
    assert response["statusCode"] == 403


class _FakeMetadataStore(ArtifactMetadataStore):
    def __init__(self) -> None:
        self.records: dict[str, Artifact] = {}

    def save(self, artifact: Artifact, *, overwrite: bool = False) -> None:
        artifact_id = artifact.metadata.id
        self.records[artifact_id] = artifact

    def load(self, artifact_id: str) -> Artifact:
        try:
            return self.records[artifact_id]
        except KeyError as exc:
            msg = f"Artifact '{artifact_id}' does not exist"
            raise ArtifactNotFound(msg) from exc


class _FakeLineageStore:
    def __init__(self, raises: bool = False) -> None:
        self.graphs: dict[str, ArtifactLineageGraph] = {}
        self.raises = raises

    def save(self, artifact_id: str, graph: ArtifactLineageGraph) -> None:
        self.graphs[artifact_id] = graph

    def load(self, artifact_id: str) -> ArtifactLineageGraph:
        if self.raises:
            raise LineageNotFound("no lineage")
        return self.graphs[artifact_id]
