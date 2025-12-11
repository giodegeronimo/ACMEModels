"""Tests for GET /artifact/model/{id}/rate handler."""

from __future__ import annotations

import json
from typing import Any, Dict, cast

import pytest

from backend.src.handlers.artifact_rate import app as handler
from src.models import Artifact, ArtifactData, ArtifactMetadata, ArtifactType
from src.storage.errors import ArtifactNotFound
from src.storage.metadata_store import ArtifactMetadataStore
from src.utils import auth


@pytest.fixture(autouse=True)
def _reset_store(monkeypatch: pytest.MonkeyPatch) -> None:
    handler._METADATA_STORE = cast(
        ArtifactMetadataStore, _FakeMetadataStore()
    )


def _event(artifact_id: str = "abc123") -> Dict[str, Any]:
    token = auth.issue_token("tester", is_admin=True)
    return {
        "pathParameters": {"id": artifact_id},
        "headers": {"X-Authorization": token},
    }


def test_get_rating_success(monkeypatch: pytest.MonkeyPatch) -> None:
    rating_payload = {
        "name": "demo",
        "category": "MODEL",
        "net_score": 1.0,
        "net_score_latency": 0.1,
        "ramp_up_time": 1.0,
        "ramp_up_time_latency": 0.1,
        "bus_factor": 1.0,
        "bus_factor_latency": 0.1,
        "performance_claims": 1.0,
        "performance_claims_latency": 0.1,
        "license": 1.0,
        "license_latency": 0.1,
        "dataset_and_code_score": 1.0,
        "dataset_and_code_score_latency": 0.1,
        "dataset_quality": 1.0,
        "dataset_quality_latency": 0.1,
        "code_quality": 1.0,
        "code_quality_latency": 0.1,
        "reproducibility": 1.0,
        "reproducibility_latency": 0.1,
        "reviewedness": 1.0,
        "reviewedness_latency": 0.1,
        "tree_score": 1.0,
        "tree_score_latency": 0.1,
        "size_score": 1.0,
        "size_score_latency": 0.1,
    }

    handler._METADATA_STORE.save(  # type: ignore[attr-defined]
        Artifact(
            metadata=ArtifactMetadata(
                name="demo",
                id="abc123",
                type=ArtifactType.MODEL,
            ),
            data=ArtifactData(url="https://example.com/model"),
        )
    )

    monkeypatch.setattr(
        handler,
        "load_rating",
        lambda artifact_id: (
            rating_payload if artifact_id == "abc123" else None
        ),
    )

    response = handler.lambda_handler(_event(), context={})
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["net_score"] == 1.0


def test_get_rating_missing() -> None:
    response = handler.lambda_handler(_event(), context={})
    assert response["statusCode"] == 404


class _FakeMetadataStore(ArtifactMetadataStore):
    def __init__(self) -> None:
        self.records: dict[str, Artifact] = {}

    def save(self, artifact: Artifact, *, overwrite: bool = False) -> None:
        artifact_id = artifact.metadata.id
        self.records[artifact_id] = artifact

    def load(self, artifact_id: str) -> Artifact:
        try:
            return self.records[artifact_id]
        except KeyError:
            raise ArtifactNotFound(
                f"Artifact '{artifact_id}' does not exist"
            )
