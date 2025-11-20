"""Secondary index for artifact names to support search operations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Protocol, Tuple

from src.models.artifacts import ArtifactMetadata, ArtifactType


@dataclass(frozen=True)
class NameIndexEntry:
    """A single searchable artifact entry."""

    artifact_id: str
    name: str
    artifact_type: ArtifactType
    readme_excerpt: str | None = None

    @property
    def normalized_name(self) -> str:
        return self.name.casefold()


class NameIndexStore(Protocol):
    """Interface implemented by name-index backends."""

    def save(self, entry: NameIndexEntry) -> None:
        """Insert or replace an index entry."""

    def delete(self, entry: NameIndexEntry) -> None:
        """Remove an entry from the index."""

    def scan(
        self,
        *,
        start_key: Any | None = None,
        limit: int | None = None,
    ) -> Tuple[List[NameIndexEntry], Any | None]:
        """Return a window of entries along with the pagination token."""


class InMemoryNameIndexStore(NameIndexStore):
    """Simple store used for tests and local runs."""

    def __init__(self) -> None:
        self._entries: dict[str, NameIndexEntry] = {}
        self._order: list[str] = []
        self._positions: dict[str, int] = {}

    def save(self, entry: NameIndexEntry) -> None:
        replaced = entry.artifact_id in self._entries
        self._entries[entry.artifact_id] = entry
        if not replaced:
            self._positions[entry.artifact_id] = len(self._order)
            self._order.append(entry.artifact_id)

    def delete(self, entry: NameIndexEntry) -> None:
        self._entries.pop(entry.artifact_id, None)

    def scan(
        self,
        *,
        start_key: Any | None = None,
        limit: int | None = None,
    ) -> Tuple[List[NameIndexEntry], Any | None]:
        start_index = -1
        if start_key is not None:
            start_index = self._positions.get(str(start_key), -1)

        collected: list[NameIndexEntry] = []
        next_key: str | None = None
        index = start_index + 1
        while index < len(self._order):
            artifact_id = self._order[index]
            entry = self._entries.get(artifact_id)
            index += 1
            if entry is None:
                continue
            collected.append(entry)
            if limit is not None and len(collected) >= limit:
                next_key = artifact_id
                break
        if limit is not None and len(collected) < limit:
            next_key = None
        return collected, next_key


class DynamoDBNameIndexStore(NameIndexStore):
    """DynamoDB-backed index keyed by normalized artifact name."""

    def __init__(
        self,
        table_name: str,
        *,
        resource: Any | None = None,
    ) -> None:
        if not table_name:
            raise ValueError("table_name must be provided")
        if resource is None:
            import boto3  # type: ignore[import-untyped]

            resource = boto3.resource("dynamodb")
        self._table = resource.Table(table_name)

    def save(self, entry: NameIndexEntry) -> None:
        item = {
            "normalized_name": entry.normalized_name,
            "artifact_id": entry.artifact_id,
            "name": entry.name,
            "artifact_type": entry.artifact_type.value,
        }
        if entry.readme_excerpt:
            item["readme_excerpt"] = entry.readme_excerpt
        self._table.put_item(Item=item)

    def delete(self, entry: NameIndexEntry) -> None:
        key = {
            "normalized_name": entry.normalized_name,
            "artifact_id": entry.artifact_id,
        }
        self._table.delete_item(Key=key)

    def scan(
        self,
        *,
        start_key: Any | None = None,
        limit: int | None = None,
    ) -> Tuple[List[NameIndexEntry], Any | None]:
        params: dict[str, Any] = {}
        if start_key is not None:
            params["ExclusiveStartKey"] = start_key
        if limit is not None:
            params["Limit"] = limit
        response = self._table.scan(**params)
        items = response.get("Items", [])
        entries = [
            NameIndexEntry(
                artifact_id=item["artifact_id"],
                name=item["name"],
                artifact_type=ArtifactType(item["artifact_type"]),
                readme_excerpt=item.get("readme_excerpt"),
            )
            for item in items
        ]
        return entries, response.get("LastEvaluatedKey")


def build_name_index_store_from_env() -> NameIndexStore:
    table_name = os.getenv("ARTIFACT_NAME_INDEX_TABLE")
    if table_name:
        return DynamoDBNameIndexStore(table_name)
    return InMemoryNameIndexStore()


def entry_from_metadata(
    metadata: ArtifactMetadata,
    *,
    readme_excerpt: str | None = None,
) -> NameIndexEntry:
    return NameIndexEntry(
        artifact_id=metadata.id,
        name=metadata.name,
        artifact_type=metadata.type,
        readme_excerpt=readme_excerpt,
    )
