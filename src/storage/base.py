"""Abstract repository interfaces for the registry backend."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from src.models import (Artifact, ArtifactAuditEntry, ArtifactLineageGraph,
                        ArtifactMetadata, ArtifactQuery)


class ArtifactRepository(Protocol):
    """Persistence contract for artifact metadata and ingest data."""

    def create(self, artifact: Artifact) -> Artifact:
        """Persist a new artifact; raise ValidationError on duplicate ids."""

    def update(self, artifact: Artifact) -> Artifact:
        """Replace an existing artifact record."""

    def get(self, artifact_id: str) -> Artifact:
        """Return the full artifact or raise ArtifactNotFound."""

    def get_metadata(self, artifact_id: str) -> ArtifactMetadata:
        """Return only metadata for listing endpoints."""

    def enumerate(
        self,
        queries: Sequence[ArtifactQuery],
        *,
        offset: int = 0,
        limit: int = 50,
    ) -> Sequence[ArtifactMetadata]:
        """
        List artifacts matching the supplied queries.

        The spec allows wildcard name ('*') to return all entries.
        """


class LineageRepository(Protocol):
    """Store and retrieve artifact lineage graphs."""

    def upsert(self, artifact_id: str, graph: ArtifactLineageGraph) -> None:
        """Create or replace lineage for an artifact."""

    def get(self, artifact_id: str) -> ArtifactLineageGraph:
        """Return the lineage graph or raise LineageNotFound."""


class AuditRepository(Protocol):
    """Append-only audit log per artifact."""

    def append(self, entry: ArtifactAuditEntry) -> None:
        """Record a new audit entry."""

    def list(
        self,
        artifact_id: str,
        *,
        limit: int = 100,
    ) -> Sequence[ArtifactAuditEntry]:
        """
        Return the latest audit entries or raise AuditLogNotFound if empty.
        """


class MetricsRepository(Protocol):
    """Optional counters used for per-route/component observability."""

    def increment(self, route: str, component: str) -> None:
        """Increment a counter identified by (route, component)."""

    def snapshot(self) -> Mapping[str, Mapping[str, int]]:
        """Return counts grouped by route and component."""
