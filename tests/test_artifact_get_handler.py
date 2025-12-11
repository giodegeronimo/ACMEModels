"""Tests for GET /artifacts/{artifact_type}/{id}."""

from __future__ import annotations

import json
from typing import Any, Dict, cast

import pytest

from backend.src.handlers.artifact_get import app as handler
from src.models.artifacts import (Artifact, ArtifactData, ArtifactMetadata,
                                  ArtifactType)
from src.storage.errors import ArtifactNotFound, ValidationError
from src.storage.metadata_store import ArtifactMetadataStore
from src.utils import auth


@pytest.fixture(autouse=True)
def _reset_store() -> None:
    handler._METADATA_STORE = cast(
        ArtifactMetadataStore, _FakeMetadataStore()
    )


def _event(
    *,
    artifact_type: str = "model",
    artifact_id: str = "abc123",
    token: str,
) -> Dict[str, Any]:
    return {
        "pathParameters": {
            "artifact_type": artifact_type,
            "id": artifact_id,
        },
        "headers": {"X-Authorization": token},
    }


def _store_artifact(
    *,
    artifact_id: str = "abc123",
    artifact_type: ArtifactType = ArtifactType.MODEL,
    url: str = "https://example.com/model",
) -> None:
    handler._METADATA_STORE.save(  # type: ignore[attr-defined]
        Artifact(
            metadata=ArtifactMetadata(
                name="demo",
                id=artifact_id,
                type=artifact_type,
            ),
            data=ArtifactData(url=url),
        )
    )


def test_get_artifact_success() -> None:
    token = auth.issue_token("tester", is_admin=True)
    _store_artifact()
    response = handler.lambda_handler(_event(token=token), context={})
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["metadata"]["id"] == "abc123"
    assert body["metadata"]["type"] == "model"
    assert body["data"]["url"] == "https://example.com/model"


def test_get_artifact_validates_type() -> None:
    token = auth.issue_token("tester", is_admin=True)
    _store_artifact(artifact_type=ArtifactType.CODE)
    response = handler.lambda_handler(
        _event(artifact_type="model", token=token),
        context={},
    )
    assert response["statusCode"] == 404


def test_get_artifact_missing() -> None:
    token = auth.issue_token("tester", is_admin=True)
    response = handler.lambda_handler(_event(token=token), context={})
    assert response["statusCode"] == 404


def test_get_artifact_invalid_id() -> None:
    token = auth.issue_token("tester", is_admin=True)
    event = _event(artifact_id="not valid!", token=token)
    response = handler.lambda_handler(event, context={})
    assert response["statusCode"] == 400


class _FakeMetadataStore(ArtifactMetadataStore):
    def __init__(self) -> None:
        self.records: dict[str, Artifact] = {}

    def save(self, artifact: Artifact, *, overwrite: bool = False) -> None:
        artifact_id = artifact.metadata.id
        if not overwrite and artifact_id in self.records:
            raise ValidationError(f"Artifact '{artifact_id}' already exists")
        self.records[artifact_id] = artifact

    def load(self, artifact_id: str) -> Artifact:
        try:
            return self.records[artifact_id]
        except KeyError as exc:
            raise ArtifactNotFound(
                f"Artifact '{artifact_id}' does not exist"
            ) from exc
