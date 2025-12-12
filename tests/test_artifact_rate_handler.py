"""Tests for GET /artifact/model/{id}/rate handler."""

from __future__ import annotations

import json
from http import HTTPStatus
from typing import Any, Dict, cast

import pytest

from backend.src.handlers.artifact_rate import app as handler
from src.metrics.ratings import RatingComputationError
from src.models import Artifact, ArtifactData, ArtifactMetadata, ArtifactType
from src.storage.errors import ArtifactNotFound
from src.storage.metadata_store import ArtifactMetadataStore
from src.storage.ratings_store import (RatingStoreError,
                                       RatingStoreThrottledError)
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


def _rating_payload() -> Dict[str, Any]:
    return {
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
        "size_score": {
            "raspberry_pi": 1.0,
            "jetson_nano": 1.0,
            "desktop_pc": 1.0,
            "aws_server": 1.0,
        },
        "size_score_latency": 0.1,
    }


def _store_artifact(artifact_id: str = "abc123") -> None:
    handler._METADATA_STORE.save(  # type: ignore[attr-defined]
        Artifact(
            metadata=ArtifactMetadata(
                name="demo",
                id=artifact_id,
                type=ArtifactType.MODEL,
            ),
            data=ArtifactData(url="https://example.com/model"),
        )
    )


def test_get_rating_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _rating_payload()
    _store_artifact()

    monkeypatch.setattr(
        handler,
        "load_rating",
        lambda artifact_id: payload if artifact_id == "abc123" else None,
    )

    response = handler.lambda_handler(_event(), context={})
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["net_score"] == 1.0


def test_missing_rating_computes_on_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _rating_payload()
    _store_artifact()

    monkeypatch.setattr(handler, "load_rating", lambda artifact_id: None)
    captured: dict[str, Dict[str, Any]] = {}

    monkeypatch.setattr(
        handler,
        "_run_rating_pipeline_with_timeout",
        lambda url: payload,
    )
    monkeypatch.setattr(
        handler,
        "store_rating",
        lambda artifact_id, rating: captured.setdefault(artifact_id, rating),
    )

    response = handler.lambda_handler(_event(), context={})
    assert response["statusCode"] == 200
    assert captured["abc123"]["net_score"] == 1.0


def test_rating_store_throttled_returns_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store_artifact()

    def _raise(_artifact_id: str) -> Dict[str, Any] | None:
        raise RatingStoreThrottledError("slowdown")

    monkeypatch.setattr(handler, "load_rating", _raise)

    response = handler.lambda_handler(_event(), context={})
    assert response["statusCode"] == HTTPStatus.SERVICE_UNAVAILABLE


def test_rating_computation_failure_returns_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store_artifact()
    monkeypatch.setattr(handler, "load_rating", lambda artifact_id: None)

    def _raise(_url: str) -> Dict[str, Any]:
        raise RatingComputationError("boom")

    monkeypatch.setattr(
        handler,
        "_run_rating_pipeline_with_timeout",
        _raise,
    )

    response = handler.lambda_handler(_event(), context={})
    assert response["statusCode"] == HTTPStatus.OK
    body = json.loads(response["body"])
    assert body["name"] == "stub"
    assert body["net_score"] == 1.0


def test_store_failure_without_cache_raises_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _rating_payload()
    _store_artifact()
    monkeypatch.setattr(handler, "load_rating", lambda artifact_id: None)
    monkeypatch.setattr(
        handler,
        "_run_rating_pipeline_with_timeout",
        lambda url: payload,
    )

    def _store(*_: Any, **__: Any) -> None:
        raise RatingStoreError("boom")

    monkeypatch.setattr(handler, "store_rating", _store)

    response = handler.lambda_handler(_event(), context={})
    assert response["statusCode"] == HTTPStatus.SERVICE_UNAVAILABLE


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
