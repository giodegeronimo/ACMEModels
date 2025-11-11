"""Lineage graph domain models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .artifacts import validate_artifact_id


@dataclass(frozen=True)
class ArtifactLineageNode:
    """Node participating in a lineage graph."""

    artifact_id: str
    name: str | None = None
    source: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        validate_artifact_id(self.artifact_id)


@dataclass(frozen=True)
class ArtifactLineageEdge:
    """Directed relationship between lineage nodes."""

    from_node_artifact_id: str
    to_node_artifact_id: str
    relationship: str

    def __post_init__(self) -> None:
        validate_artifact_id(self.from_node_artifact_id)
        validate_artifact_id(self.to_node_artifact_id)
        if not self.relationship:
            raise ValueError("Lineage relationship label cannot be empty")


@dataclass(frozen=True)
class ArtifactLineageGraph:
    """Complete lineage graph for a single artifact."""

    nodes: Sequence[ArtifactLineageNode]
    edges: Sequence[ArtifactLineageEdge]

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("Lineage graph must include at least one node")
