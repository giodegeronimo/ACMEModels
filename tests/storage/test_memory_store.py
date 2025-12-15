from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.models.artifacts import (Artifact, ArtifactData, ArtifactMetadata,
                                  ArtifactQuery, ArtifactType)
from src.models.audit import ArtifactAuditAction, ArtifactAuditEntry, User
from src.models.lineage import (ArtifactLineageEdge, ArtifactLineageGraph,
                                ArtifactLineageNode)
from src.storage.errors import (ArtifactNotFound, AuditLogNotFound,
                                LineageNotFound, ValidationError)
from src.storage.memory import (InMemoryArtifactRepository,
                                InMemoryAuditRepository,
                                InMemoryLineageRepository)


def _artifact(artifact_id: str, name: str = "a") -> Artifact:
    return Artifact(
        metadata=ArtifactMetadata(
            name=name,
            id=artifact_id,
            type=ArtifactType.MODEL,
        ),
        data=ArtifactData(url="https://example.com/x"),
    )


def test_in_memory_artifact_repository_create_update_get_and_enumerate() -> None:
    repo = InMemoryArtifactRepository()

    repo.create(_artifact("id-1", name="n"))
    with pytest.raises(ValidationError, match="already exists"):
        repo.create(_artifact("id-1", name="n"))

    with pytest.raises(ArtifactNotFound):
        repo.update(_artifact("missing"))

    updated = _artifact("id-1", name="n2")
    repo.update(updated)
    assert repo.get("id-1").metadata.name == "n2"
    assert repo.get_metadata("id-1").id == "id-1"
    with pytest.raises(ArtifactNotFound):
        repo.get("missing")

    repo.create(_artifact("id-2", name="n"))
    all_items = repo.enumerate([])
    assert len(all_items) == 2

    by_name = repo.enumerate([ArtifactQuery(name="n")])
    assert {item.id for item in by_name} == {"id-2"}

    with pytest.raises(ValidationError, match="non-negative"):
        repo.enumerate([], offset=-1)


def test_in_memory_lineage_repository_upsert_and_get_validation() -> None:
    repo = InMemoryLineageRepository()
    node = ArtifactLineageNode(artifact_id="id-1", name="model")
    graph = ArtifactLineageGraph(nodes=[node], edges=[])
    repo.upsert("id-1", graph)
    assert repo.get("id-1").nodes[0].artifact_id == "id-1"

    with pytest.raises(LineageNotFound):
        repo.get("missing")

    bad_edge = ArtifactLineageEdge(
        from_node_artifact_id="id-1",
        to_node_artifact_id="missing",
        relationship="depends_on",
    )
    bad_graph = ArtifactLineageGraph(nodes=[node], edges=[bad_edge])
    with pytest.raises(ValidationError, match="missing node"):
        repo.upsert("id-1", bad_graph)


def test_in_memory_audit_repository_append_and_list() -> None:
    repo = InMemoryAuditRepository()
    entry = ArtifactAuditEntry(
        user=User(name="tester", is_admin=True),
        date=datetime.now(timezone.utc),
        artifact=_artifact("id-1").metadata,
        action=ArtifactAuditAction.CREATE,
    )
    repo.append(entry)
    assert len(repo.list("id-1")) == 1
    with pytest.raises(AuditLogNotFound):
        repo.list("missing")
