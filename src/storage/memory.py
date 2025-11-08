"""In-memory repository implementations for development and tests."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

from src.models import (Artifact, ArtifactAuditEntry, ArtifactLineageGraph,
                        ArtifactMetadata, ArtifactQuery)

from .base import (ArtifactRepository, AuditRepository, LineageRepository,
                   MetricsRepository)
from .errors import (ArtifactNotFound, AuditLogNotFound, LineageNotFound,
                     ValidationError)


def _artifact_matches_query(
    metadata: ArtifactMetadata, query: ArtifactQuery
) -> bool:
    if query.name != "*" and metadata.name != query.name:
        return False
    if query.types:
        allowed = set(query.types)
        if metadata.type not in allowed:
            return False
    return True


class InMemoryArtifactRepository(ArtifactRepository):
    """Dictionary-backed artifact store."""

    def __init__(self) -> None:
        self._artifacts: Dict[str, Artifact] = {}

    def create(self, artifact: Artifact) -> Artifact:
        artifact_id = artifact.metadata.id
        if artifact_id in self._artifacts:
            raise ValidationError(f"Artifact '{artifact_id}' already exists")
        self._artifacts[artifact_id] = artifact
        return artifact

    def update(self, artifact: Artifact) -> Artifact:
        artifact_id = artifact.metadata.id
        if artifact_id not in self._artifacts:
            raise ArtifactNotFound(f"Artifact '{artifact_id}' does not exist")
        self._artifacts[artifact_id] = artifact
        return artifact

    def get(self, artifact_id: str) -> Artifact:
        try:
            return self._artifacts[artifact_id]
        except KeyError as exc:
            raise ArtifactNotFound(
                f"Artifact '{artifact_id}' does not exist"
            ) from exc

    def get_metadata(self, artifact_id: str) -> ArtifactMetadata:
        return self.get(artifact_id).metadata

    def enumerate(
        self,
        queries: Sequence[ArtifactQuery],
        *,
        offset: int = 0,
        limit: int = 50,
    ) -> Sequence[ArtifactMetadata]:
        if limit < 0 or offset < 0:
            raise ValidationError("offset and limit must be non-negative")

        artifacts = list(self._artifacts.values())
        matched: List[Artifact]
        if not queries:
            matched = list(artifacts)
        else:
            seen_ids: set[str] = set()
            matched = []
            for query in queries:
                for artifact in artifacts:
                    artifact_id = artifact.metadata.id
                    if artifact_id in seen_ids:
                        continue
                    if _artifact_matches_query(artifact.metadata, query):
                        matched.append(artifact)
                        seen_ids.add(artifact_id)
        matched.sort(key=lambda a: (a.metadata.name, a.metadata.id))
        slice_start = offset
        slice_end = offset + limit if limit else None
        return [
            artifact.metadata for artifact in matched[slice_start:slice_end]
        ]


class InMemoryLineageRepository(LineageRepository):
    """Store lineage graphs keyed by artifact id."""

    def __init__(self) -> None:
        self._graphs: Dict[str, ArtifactLineageGraph] = {}

    @staticmethod
    def _validate_graph(graph: ArtifactLineageGraph) -> None:
        node_ids = {node.artifact_id for node in graph.nodes}
        for edge in graph.edges:
            if edge.from_node_artifact_id not in node_ids:
                raise ValidationError(
                    "Edge references missing node "
                    f"{edge.from_node_artifact_id}"
                )
            if edge.to_node_artifact_id not in node_ids:
                raise ValidationError(
                    "Edge references missing node "
                    f"{edge.to_node_artifact_id}"
                )

    def upsert(self, artifact_id: str, graph: ArtifactLineageGraph) -> None:
        self._validate_graph(graph)
        self._graphs[artifact_id] = graph

    def get(self, artifact_id: str) -> ArtifactLineageGraph:
        try:
            return self._graphs[artifact_id]
        except KeyError as exc:
            raise LineageNotFound(
                f"Lineage for '{artifact_id}' not found"
            ) from exc


class InMemoryAuditRepository(AuditRepository):
    """Append-only audit log stored in memory."""

    def __init__(self) -> None:
        self._entries: Dict[str, List[ArtifactAuditEntry]]
        self._entries = defaultdict(list)

    def append(self, entry: ArtifactAuditEntry) -> None:
        artifact_id = entry.artifact.id
        self._entries[artifact_id].append(entry)

    def list(
        self,
        artifact_id: str,
        *,
        limit: int = 100,
    ) -> Sequence[ArtifactAuditEntry]:
        events = self._entries.get(artifact_id)
        if not events:
            raise AuditLogNotFound(
                f"No audit entries recorded for '{artifact_id}'"
            )
        ordered = list(reversed(events))
        if limit:
            ordered = ordered[:limit]
        return ordered


class InMemoryMetricsRepository(MetricsRepository):
    """Route/component counter store."""

    def __init__(self) -> None:
        self._counters: Dict[str, Dict[str, int]]
        self._counters = defaultdict(dict)

    def increment(self, route: str, component: str) -> None:
        if not route or not component:
            raise ValidationError("route and component must be provided")
        component_counts = self._counters.setdefault(route, {})
        component_counts[component] = component_counts.get(component, 0) + 1

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        return {
            route: dict(components)
            for route, components in self._counters.items()
        }
