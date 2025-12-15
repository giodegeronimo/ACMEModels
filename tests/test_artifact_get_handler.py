"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for GET /artifacts/{artifact_type}/{id}.
"""

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
    """
    _reset_store: Function description.
    :param:
    :returns:
    """

    handler._METADATA_STORE = cast(
        ArtifactMetadataStore, _FakeMetadataStore()
    )


def _event(
    *,
    artifact_type: str = "model",
    artifact_id: str = "abc123",
    token: str,
) -> Dict[str, Any]:
    """
    _event: Function description.
    :param artifact_type:
    :param artifact_id:
    :param token:
    :returns:
    """

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
    """
    _store_artifact: Function description.
    :param artifact_id:
    :param artifact_type:
    :param url:
    :returns:
    """

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
    """
    test_get_artifact_success: Function description.
    :param:
    :returns:
    """

    token = auth.issue_token("tester", is_admin=True)
    _store_artifact()
    response = handler.lambda_handler(_event(token=token), context={})
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["metadata"]["id"] == "abc123"
    assert body["metadata"]["type"] == "model"
    assert body["data"]["url"] == "https://example.com/model"


def test_get_artifact_validates_type() -> None:
    """
    test_get_artifact_validates_type: Function description.
    :param:
    :returns:
    """

    token = auth.issue_token("tester", is_admin=True)
    _store_artifact(artifact_type=ArtifactType.CODE)
    response = handler.lambda_handler(
        _event(artifact_type="model", token=token),
        context={},
    )
    assert response["statusCode"] == 404


def test_get_artifact_missing() -> None:
    """
    test_get_artifact_missing: Function description.
    :param:
    :returns:
    """

    token = auth.issue_token("tester", is_admin=True)
    response = handler.lambda_handler(_event(token=token), context={})
    assert response["statusCode"] == 404


def test_get_artifact_invalid_id() -> None:
    """
    test_get_artifact_invalid_id: Function description.
    :param:
    :returns:
    """

    token = auth.issue_token("tester", is_admin=True)
    event = _event(artifact_id="not valid!", token=token)
    response = handler.lambda_handler(event, context={})
    assert response["statusCode"] == 400


class _FakeMetadataStore(ArtifactMetadataStore):
    """
    _FakeMetadataStore: Class description.
    """

    def __init__(self) -> None:
        """
        __init__: Function description.
        :param:
        :returns:
        """

        self.records: dict[str, Artifact] = {}

    def save(self, artifact: Artifact, *, overwrite: bool = False) -> None:
        """
        save: Function description.
        :param artifact:
        :param overwrite:
        :returns:
        """

        artifact_id = artifact.metadata.id
        if not overwrite and artifact_id in self.records:
            raise ValidationError(f"Artifact '{artifact_id}' already exists")
        self.records[artifact_id] = artifact

    def load(self, artifact_id: str) -> Artifact:
        """
        load: Function description.
        :param artifact_id:
        :returns:
        """

        try:
            return self.records[artifact_id]
        except KeyError as exc:
            raise ArtifactNotFound(
                f"Artifact '{artifact_id}' does not exist"
            ) from exc
