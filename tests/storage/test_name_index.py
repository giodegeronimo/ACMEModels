"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for the artifact name index stores.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.models.artifacts import ArtifactMetadata, ArtifactType
from src.storage.name_index import (DynamoDBNameIndexStore,
                                    InMemoryNameIndexStore, NameIndexEntry,
                                    entry_from_metadata)


def _metadata(name: str, artifact_id: str = "artifact-1") -> ArtifactMetadata:
    """
    _metadata: Function description.
    :param name:
    :param artifact_id:
    :returns:
    """

    return ArtifactMetadata(name=name, id=artifact_id, type=ArtifactType.MODEL)


def _entry(
    name: str,
    artifact_id: str = "artifact-1",
    readme: str | None = None,
) -> NameIndexEntry:
    """
    _entry: Function description.
    :param name:
    :param artifact_id:
    :param readme:
    :returns:
    """

    return entry_from_metadata(
        _metadata(name=name, artifact_id=artifact_id),
        readme_excerpt=readme,
    )


def test_in_memory_store_persists_and_scans_entries() -> None:
    """
    test_in_memory_store_persists_and_scans_entries: Function description.
    :param:
    :returns:
    """

    store = InMemoryNameIndexStore()
    first = _entry("FirstModel", "a1")
    second = _entry("SecondModel", "a2")

    store.save(first)
    store.save(second)

    entries, token = store.scan()
    assert entries == [first, second]
    assert token is None


def test_in_memory_store_supports_deletion_and_pagination() -> None:
    """
    test_in_memory_store_supports_deletion_and_pagination: Function description.
    :param:
    :returns:
    """

    store = InMemoryNameIndexStore()
    entries = [_entry(f"Model{i}", f"id-{i}") for i in range(3)]
    for entry in entries:
        store.save(entry)

    first_page, token = store.scan(limit=2)
    assert first_page == entries[:2]
    assert token == entries[1].artifact_id

    store.delete(entries[1])

    second_page, _ = store.scan(start_key=token)
    assert second_page == entries[2:]


class _FakeTable:
    """
    _FakeTable: Class description.
    """

    def __init__(self) -> None:
        """
        __init__: Function description.
        :param:
        :returns:
        """

        self.put_items: List[Dict[str, Any]] = []
        self.delete_keys: List[Dict[str, Any]] = []
        self.scan_calls: List[Dict[str, Any]] = []
        self.scan_responses: List[Dict[str, Any]] = []

    def put_item(self, Item: Dict[str, Any]) -> None:  # noqa: N803
        """
        put_item: Function description.
        :param Item:
        :returns:
        """

        self.put_items.append(Item)

    def delete_item(self, Key: Dict[str, Any]) -> None:  # noqa: N803
        """
        delete_item: Function description.
        :param Key:
        :returns:
        """

        self.delete_keys.append(Key)

    def scan(self, **kwargs: Any) -> Dict[str, Any]:
        """
        scan: Function description.
        :param **kwargs:
        :returns:
        """

        self.scan_calls.append(kwargs)
        if self.scan_responses:
            return self.scan_responses.pop(0)
        return {"Items": []}


class _FakeResource:
    """
    _FakeResource: Class description.
    """

    def __init__(self, table: _FakeTable) -> None:
        """
        __init__: Function description.
        :param table:
        :returns:
        """

        self._table = table
        self.requested_table: Optional[str] = None

    def Table(self, name: str) -> _FakeTable:  # noqa: N802
        """
        Table: Function description.
        :param name:
        :returns:
        """

        self.requested_table = name
        return self._table


def test_dynamodb_store_writes_expected_items() -> None:
    """
    test_dynamodb_store_writes_expected_items: Function description.
    :param:
    :returns:
    """

    table = _FakeTable()
    resource = _FakeResource(table)
    store = DynamoDBNameIndexStore("NameIndex", resource=resource)
    entry = _entry("SampleName", "s1", readme="README content")

    store.save(entry)
    assert resource.requested_table == "NameIndex"
    assert table.put_items == [
        {
            "normalized_name": entry.name.casefold(),
            "artifact_id": entry.artifact_id,
            "name": entry.name,
            "artifact_type": entry.artifact_type.value,
            "readme_excerpt": "README content",
        }
    ]

    store.delete(entry)
    assert table.delete_keys == [
        {
            "normalized_name": entry.name.casefold(),
            "artifact_id": entry.artifact_id,
        }
    ]


def test_dynamodb_store_scan_returns_entries() -> None:
    """
    test_dynamodb_store_scan_returns_entries: Function description.
    :param:
    :returns:
    """

    table = _FakeTable()
    table.scan_responses.append(
        {
            "Items": [
                {
                    "normalized_name": "model",
                    "artifact_id": "abc",
                    "name": "Model",
                    "artifact_type": "model",
                    "readme_excerpt": "Details",
                }
            ],
            "LastEvaluatedKey": {
                "artifact_id": "abc",
                "normalized_name": "model",
            },
        }
    )
    store = DynamoDBNameIndexStore("table", resource=_FakeResource(table))

    entries, token = store.scan(limit=5, start_key={"artifact_id": "xyz"})
    assert len(entries) == 1
    assert entries[0].artifact_id == "abc"
    assert entries[0].readme_excerpt == "Details"
    assert token == {"artifact_id": "abc", "normalized_name": "model"}
    assert table.scan_calls == [
        {"ExclusiveStartKey": {"artifact_id": "xyz"}, "Limit": 5}
    ]


def test_dynamodb_store_requires_table_name() -> None:
    """
    test_dynamodb_store_requires_table_name: Function description.
    :param:
    :returns:
    """

    with pytest.raises(ValueError, match="table_name must be provided"):
        DynamoDBNameIndexStore("")
