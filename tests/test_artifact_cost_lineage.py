"""Integration test: ensure lineage flows into artifact cost calculation."""

from __future__ import annotations

import json
import sys
import types
from typing import Any, Dict, cast

import pytest

from backend.src.handlers.artifact_cost import app as cost_handler
from src.models.artifacts import (Artifact, ArtifactData, ArtifactMetadata,
                                  ArtifactType)
from src.models.lineage import (ArtifactLineageEdge, ArtifactLineageGraph,
                                ArtifactLineageNode)
from src.storage.lineage_store import LineageStore
from src.storage.memory import get_lineage_repo
from src.storage.metadata_store import ArtifactMetadataStore


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BUCKET", "test-bucket")
    monkeypatch.setenv("ARTIFACT_STORAGE_PREFIX", "artifacts")
    # ensure metadata store is fresh
    cost_handler._METADATA_STORE = cast(
        ArtifactMetadataStore, _FakeMetadataStore()
    )
    cost_handler._LINEAGE_STORE = cast(
        LineageStore, _FakeLineageStore()
    )
    # clear in-memory lineage repo
    repo = get_lineage_repo()
    repo._graphs.clear()


class _FakeMetadataStore(ArtifactMetadataStore):
    def __init__(self) -> None:
        self.records: dict[str, Artifact] = {}

    def save(self, artifact: Artifact, *, overwrite: bool = False) -> None:
        self.records[artifact.metadata.id] = artifact

    def load(self, artifact_id: str) -> Artifact:
        return self.records[artifact_id]


class _FakeLineageStore:
    def __init__(self) -> None:
        self.graphs: dict[str, ArtifactLineageGraph] = {}

    def save(self, artifact_id: str, graph: ArtifactLineageGraph) -> None:
        self.graphs[artifact_id] = graph

    def load(self, artifact_id: str) -> ArtifactLineageGraph:
        return self.graphs[artifact_id]


class _FakeS3Client:
    def __init__(self, sizes: Dict[str, int]):
        # sizes keyed by Key (prefix/artifact_id)
        self._sizes = sizes

    def head_object(self, Bucket: str, Key: str):
        # Return content length for known keys
        if Key in self._sizes:
            return {"ContentLength": self._sizes[Key]}
        raise Exception("No such key")


def _event(artifact_id: str, dependency: bool = True) -> Dict[str, Any]:
    query = {"dependency": "true"} if dependency else {}
    return {
        "pathParameters": {"artifact_type": "model", "id": artifact_id},
        "queryStringParameters": query,
        "headers": {"X-Authorization": "token"},
    }


def test_cost_includes_lineage_dependencies(monkeypatch) -> None:
    # Prepare artifact metadata
    root_id = "root123"
    dep_id = "dep456"
    cost_handler._METADATA_STORE.save(  # type: ignore[attr-defined]
        Artifact(
            metadata=ArtifactMetadata(
                name="root",
                id=root_id,
                type=ArtifactType.MODEL,
            ),
            data=ArtifactData(url="https://example.com/root"),
        )
    )

    # Upsert lineage graph: dep -> root
    nodes = [
        ArtifactLineageNode(artifact_id=root_id),
        ArtifactLineageNode(artifact_id=dep_id),
    ]
    edges = [
        ArtifactLineageEdge(
            from_node_artifact_id=dep_id,
            to_node_artifact_id=root_id,
            relationship="depends_on",
        )
    ]
    graph = ArtifactLineageGraph(nodes=nodes, edges=edges)
    get_lineage_repo().upsert(root_id, graph)
    cost_handler._LINEAGE_STORE.save(  # type: ignore[attr-defined]
        root_id, graph
    )

    # Mock boto3 module to provide S3 client with known sizes
    sizes = {
        f"artifacts/{root_id}": 1048576,  # 1 MB
        f"artifacts/{dep_id}": 2097152,  # 2 MB
    }
    fake_boto3 = types.ModuleType("boto3")
    cast(Any, fake_boto3).client = (
        lambda service, **kwargs: _FakeS3Client(sizes)
    )
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    # Call the cost handler with dependency param
    response = cost_handler.lambda_handler(
        _event(root_id, dependency=True), context={}
    )
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    # Expect both root and dep in the result
    assert root_id in body
    assert dep_id in body
    # Check numeric values: standalone and total for root should include dep
    root_info = body[root_id]
    dep_info = body[dep_id]
    assert root_info["standalone_cost"] == 1.0
    assert dep_info["standalone_cost"] == 2.0
    assert root_info["total_cost"] == pytest.approx(3.0, rel=1e-6)
