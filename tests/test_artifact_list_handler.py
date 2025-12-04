"""Tests for POST /artifacts listing handler."""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import pytest

from backend.src.handlers.artifact_list import app as handler
from src.models import Artifact, ArtifactData, ArtifactMetadata, ArtifactType
from src.storage.errors import ArtifactNotFound
from src.storage.name_index import InMemoryNameIndexStore, entry_from_metadata
from src.utils import auth


@pytest.fixture(autouse=True)
def _patch_stores(monkeypatch: pytest.MonkeyPatch) -> Tuple[
    InMemoryNameIndexStore, "_FakeMetadataStore"
]:
    index = InMemoryNameIndexStore()
    meta_store = _FakeMetadataStore()
    monkeypatch.setattr(handler, "_NAME_INDEX", index)
    monkeypatch.setattr(handler, "_METADATA_STORE", meta_store)
    monkeypatch.setattr(handler, "MAX_RESULTS", 2)
    return index, meta_store


def _event(body: Any, *, offset: str | None = None) -> Dict[str, Any]:
    payload = body if isinstance(body, str) else json.dumps(body)
    token = auth.issue_token("tester", is_admin=True)
    event: Dict[str, Any] = {
        "headers": {"X-Authorization": token},
        "body": payload,
        "queryStringParameters": {"offset": offset} if offset else None,
    }
    return event


def _store_artifact(
    index: InMemoryNameIndexStore,
    store: "_FakeMetadataStore",
    *,
    artifact_id: str,
    name: str,
    artifact_type: ArtifactType,
) -> None:
    metadata = ArtifactMetadata(name=name, id=artifact_id, type=artifact_type)
    artifact = Artifact(metadata=metadata, data=ArtifactData(url="https://x"))
    store.records[artifact_id] = artifact
    index.save(entry_from_metadata(metadata))


def test_list_all_artifacts_returns_paginated_results(
    _patch_stores: Tuple[InMemoryNameIndexStore, "_FakeMetadataStore"]
) -> None:
    index, store = _patch_stores
    _store_artifact(index, store, artifact_id="a1", name="alpha",
                    artifact_type=ArtifactType.MODEL)
    _store_artifact(index, store, artifact_id="a2", name="beta",
                    artifact_type=ArtifactType.DATASET)
    _store_artifact(index, store, artifact_id="a3", name="gamma",
                    artifact_type=ArtifactType.CODE)

    response = handler.lambda_handler(_event([{"name": "*"}]), {})
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert len(body) == 2
    assert "offset" in response["headers"]
    next_offset = response["headers"]["offset"]

    second = handler.lambda_handler(
        _event([{"name": "*"}], offset=next_offset),
        {},
    )
    assert second["statusCode"] == 200
    body2 = json.loads(second["body"])
    assert len(body2) == 1
    assert "offset" not in second["headers"]


def test_filters_by_name_and_type(
    _patch_stores: Tuple[InMemoryNameIndexStore, "_FakeMetadataStore"]
) -> None:
    index, store = _patch_stores
    _store_artifact(index, store, artifact_id="a1", name="alpha",
                    artifact_type=ArtifactType.MODEL)
    _store_artifact(index, store, artifact_id="a2", name="alpha",
                    artifact_type=ArtifactType.CODE)
    _store_artifact(index, store, artifact_id="a3", name="beta",
                    artifact_type=ArtifactType.DATASET)

    response = handler.lambda_handler(
        _event([{"name": "alpha", "types": ["model"]}]),
        {},
    )
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert len(body) == 1
    assert body[0]["name"] == "alpha"
    assert body[0]["type"] == "model"


def test_invalid_body_returns_400() -> None:
    response = handler.lambda_handler(_event({"name": "*"}), {})
    assert response["statusCode"] == 400


def test_invalid_offset_returns_400() -> None:
    response = handler.lambda_handler(
        _event([{"name": "*"}], offset="@@@"),
        {},
    )
    assert response["statusCode"] == 400


class _FakeMetadataStore:
    def __init__(self) -> None:
        self.records: Dict[str, Artifact] = {}

    def load(self, artifact_id: str) -> Artifact:
        if artifact_id not in self.records:
            raise ArtifactNotFound(f"{artifact_id} missing")
        return self.records[artifact_id]
