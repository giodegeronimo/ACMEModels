"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for the in-memory repository implementations.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.models import (Artifact, ArtifactAuditAction, ArtifactAuditEntry,
                        ArtifactData, ArtifactLineageEdge,
                        ArtifactLineageGraph, ArtifactLineageNode,
                        ArtifactMetadata, ArtifactQuery, ArtifactType, User)
from src.storage.errors import (ArtifactNotFound, AuditLogNotFound,
                                LineageNotFound, ValidationError)
from src.storage.memory import (InMemoryArtifactRepository,
                                InMemoryAuditRepository,
                                InMemoryLineageRepository,
                                InMemoryMetricsRepository)


def _make_artifact(
    artifact_id: str, name: str, type_: ArtifactType
) -> Artifact:
    """
    _make_artifact: Function description.
    :param artifact_id:
    :param name:
    :param type_:
    :returns:
    """

    metadata = ArtifactMetadata(name=name, id=artifact_id, type=type_)
    data = ArtifactData(url=f"https://example.com/{artifact_id}.zip")
    return Artifact(metadata=metadata, data=data)


def test_artifact_repository_create_and_enumerate() -> None:
    """
    test_artifact_repository_create_and_enumerate: Function description.
    :param:
    :returns:
    """

    repo = InMemoryArtifactRepository()
    repo.create(_make_artifact("id-1", "alpha", ArtifactType.MODEL))
    repo.create(_make_artifact("id-2", "beta", ArtifactType.DATASET))

    fetched = repo.get("id-1")
    assert fetched.metadata.name == "alpha"

    everything = repo.enumerate([], limit=10)
    assert [meta.id for meta in everything] == ["id-1", "id-2"]

    filtered = repo.enumerate(
        [ArtifactQuery(name="beta", types=[ArtifactType.DATASET])]
    )
    assert len(filtered) == 1 and filtered[0].name == "beta"

    wildcard = repo.enumerate([ArtifactQuery(name="*")], limit=1)
    assert len(wildcard) == 1


def test_artifact_repository_update_and_duplicates() -> None:
    """
    test_artifact_repository_update_and_duplicates: Function description.
    :param:
    :returns:
    """

    repo = InMemoryArtifactRepository()
    artifact = _make_artifact("id-1", "alpha", ArtifactType.MODEL)
    repo.create(artifact)

    with pytest.raises(ValidationError):
        repo.create(artifact)

    updated = _make_artifact("id-1", "alpha-v2", ArtifactType.MODEL)
    repo.update(updated)
    assert repo.get_metadata("id-1").name == "alpha-v2"

    with pytest.raises(ArtifactNotFound):
        repo.update(_make_artifact("missing", "ghost", ArtifactType.CODE))


def test_lineage_repository_validates_edges() -> None:
    """
    test_lineage_repository_validates_edges: Function description.
    :param:
    :returns:
    """

    repo = InMemoryLineageRepository()
    nodes = [
        ArtifactLineageNode(artifact_id="root", name="root"),
        ArtifactLineageNode(artifact_id="child", name="child"),
    ]
    edges = [
        ArtifactLineageEdge(
            from_node_artifact_id="root",
            to_node_artifact_id="child",
            relationship="parent",
        )
    ]
    graph = ArtifactLineageGraph(nodes=nodes, edges=edges)
    repo.upsert("root", graph)
    assert repo.get("root").nodes == nodes

    bad_edge = ArtifactLineageEdge(
        from_node_artifact_id="root",
        to_node_artifact_id="ghost",
        relationship="invalid",
    )
    bad_graph = ArtifactLineageGraph(nodes=nodes, edges=[bad_edge])
    with pytest.raises(ValidationError):
        repo.upsert("root", bad_graph)

    with pytest.raises(LineageNotFound):
        repo.get("missing")


def test_audit_repository_orders_latest_first() -> None:
    """
    test_audit_repository_orders_latest_first: Function description.
    :param:
    :returns:
    """

    repo = InMemoryAuditRepository()
    user = User(name="Casey", is_admin=False)
    artifact = ArtifactMetadata(
        name="alpha",
        id="id-1",
        type=ArtifactType.MODEL,
    )

    first = ArtifactAuditEntry(
        user=user,
        date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        artifact=artifact,
        action=ArtifactAuditAction.CREATE,
    )
    second = ArtifactAuditEntry(
        user=user,
        date=datetime(2024, 1, 2, tzinfo=timezone.utc),
        artifact=artifact,
        action=ArtifactAuditAction.UPDATE,
    )
    repo.append(first)
    repo.append(second)

    results = repo.list("id-1", limit=1)
    assert results[0].action == ArtifactAuditAction.UPDATE

    with pytest.raises(AuditLogNotFound):
        repo.list("missing")


def test_metrics_repository_counts() -> None:
    """
    test_metrics_repository_counts: Function description.
    :param:
    :returns:
    """

    repo = InMemoryMetricsRepository()
    repo.increment("/health", "lambda")
    repo.increment("/health", "lambda")
    repo.increment("/artifact", "db")

    snapshot = repo.snapshot()
    assert snapshot["/health"]["lambda"] == 2
    assert snapshot["/artifact"]["db"] == 1

    with pytest.raises(ValidationError):
        repo.increment("", "component")


def test_artifact_repository_enumerate_rejects_negative_paging() -> None:
    """
    test_artifact_repository_enumerate_rejects_negative_paging: Function description.
    :param:
    :returns:
    """

    repo = InMemoryArtifactRepository()
    with pytest.raises(ValidationError, match="offset and limit must be non-negative"):
        repo.enumerate([], offset=-1, limit=1)
