"""Tests for DELETE /artifacts/{artifact_type}/{id} handler."""

from __future__ import annotations

import json
from typing import Any, Dict, cast
from unittest.mock import MagicMock

import pytest

from backend.src.handlers.artifact_delete import app as handler
from src.models import Artifact, ArtifactData, ArtifactMetadata, ArtifactType
from src.storage.errors import ArtifactNotFound
from src.utils import auth


@pytest.fixture(autouse=True)
def _reset_stores(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module-level stores and S3 client before each test."""
    handler._METADATA_STORE = cast(Any, _FakeMetadataStore())
    handler._NAME_INDEX = cast(Any, _FakeNameIndex())
    handler._S3_CLIENT = MagicMock()

    # Set required environment variables
    monkeypatch.setenv("ARTIFACT_STORAGE_BUCKET", "test-bucket")
    monkeypatch.setenv("MODEL_RESULTS_BUCKET", "test-results-bucket")


def _event(
    artifact_type: str = "model", artifact_id: str = "test123"
) -> Dict[str, Any]:
    token = auth.issue_token("tester", is_admin=True)
    return {
        "pathParameters": {"artifact_type": artifact_type, "id": artifact_id},
        "headers": {"X-Authorization": token},
    }


def test_delete_model_success() -> None:
    """Test successful deletion of a model artifact."""
    artifact_id = "test123"
    artifact = Artifact(
        metadata=ArtifactMetadata(
            name="test-model",
            id=artifact_id,
            type=ArtifactType.MODEL,
        ),
        data=ArtifactData(url="https://example.com/model"),
    )
    handler._METADATA_STORE.save(artifact)

    response = handler.lambda_handler(_event("model", artifact_id), context={})

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["message"] == "Artifact deleted successfully"

    # Verify all S3 deletions were called: blob, metadata, rating
    s3_client = handler._S3_CLIENT
    assert s3_client.delete_object.call_count == 3  # type: ignore[union-attr]


def test_delete_dataset_success() -> None:
    """Test successful deletion of a dataset artifact (no rating)."""
    artifact_id = "dataset456"
    artifact = Artifact(
        metadata=ArtifactMetadata(
            name="test-dataset",
            id=artifact_id,
            type=ArtifactType.DATASET,
        ),
        data=ArtifactData(url="https://example.com/dataset"),
    )
    handler._METADATA_STORE.save(artifact)

    response = handler.lambda_handler(
        _event("dataset", artifact_id), context={}
    )

    assert response["statusCode"] == 200
    # Verify rating deletion was NOT called (only blob and metadata)
    s3_client = handler._S3_CLIENT
    assert s3_client.delete_object.call_count == 2  # type: ignore[union-attr]


def test_delete_artifact_not_found() -> None:
    """Test deletion of non-existent artifact returns 404."""
    response = handler.lambda_handler(
        _event("model", "nonexistent"), context={}
    )

    assert response["statusCode"] == 404
    body = json.loads(response["body"])
    error_msg = body["error"].lower()
    assert "does not exist" in error_msg or "not found" in error_msg


def test_delete_artifact_type_mismatch() -> None:
    """Test deletion fails when artifact type doesn't match."""
    artifact_id = "test123"
    artifact = Artifact(
        metadata=ArtifactMetadata(
            name="test-model",
            id=artifact_id,
            type=ArtifactType.MODEL,
        ),
        data=ArtifactData(url="https://example.com/model"),
    )
    handler._METADATA_STORE.save(artifact)

    # Try to delete as dataset when it's actually a model
    response = handler.lambda_handler(
        _event("dataset", artifact_id), context={}
    )

    assert response["statusCode"] == 404
    body = json.loads(response["body"])
    assert "not found" in body["error"].lower()


def test_delete_partial_failure_blob() -> None:
    """Test that blob deletion failure results in 500."""
    artifact_id = "test123"
    artifact = Artifact(
        metadata=ArtifactMetadata(
            name="test-model",
            id=artifact_id,
            type=ArtifactType.MODEL,
        ),
        data=ArtifactData(url="https://example.com/model"),
    )
    handler._METADATA_STORE.save(artifact)

    # Make blob deletion fail
    s3_client = handler._S3_CLIENT
    s3_client.delete_object.side_effect = [  # type: ignore[union-attr]
        Exception("S3 blob error"),
        None,  # metadata succeeds
        None,  # rating succeeds
    ]

    response = handler.lambda_handler(_event("model", artifact_id), context={})

    assert response["statusCode"] == 500
    body = json.loads(response["body"])
    assert "deletion incomplete" in body["error"].lower()
    assert "blob" in body["error"]


def test_delete_partial_failure_metadata() -> None:
    """Test that metadata deletion failure results in 500."""
    artifact_id = "test123"
    artifact = Artifact(
        metadata=ArtifactMetadata(
            name="test-model",
            id=artifact_id,
            type=ArtifactType.MODEL,
        ),
        data=ArtifactData(url="https://example.com/model"),
    )
    handler._METADATA_STORE.save(artifact)

    # Make metadata deletion fail
    s3_client = handler._S3_CLIENT
    s3_client.delete_object.side_effect = [  # type: ignore[union-attr]
        None,  # blob succeeds
        Exception("S3 metadata error"),
        None,  # rating succeeds
    ]

    response = handler.lambda_handler(_event("model", artifact_id), context={})

    assert response["statusCode"] == 500
    body = json.loads(response["body"])
    assert "deletion incomplete" in body["error"].lower()
    assert "metadata" in body["error"]


def test_delete_multiple_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that multiple deletion failures are all reported."""
    artifact_id = "test123"
    artifact = Artifact(
        metadata=ArtifactMetadata(
            name="test-model",
            id=artifact_id,
            type=ArtifactType.MODEL,
        ),
        data=ArtifactData(url="https://example.com/model"),
    )
    handler._METADATA_STORE.save(artifact)

    # Ensure MODEL_RESULTS_BUCKET is set so rating deletion is attempted
    monkeypatch.setenv("MODEL_RESULTS_BUCKET", "test-results-bucket")

    # Make all S3 deletions fail
    s3_client = handler._S3_CLIENT
    s3_client.delete_object.side_effect = Exception("S3 error")  # type: ignore

    # Make name index deletion fail
    cast(Any, handler._NAME_INDEX).should_fail = True

    response = handler.lambda_handler(_event("model", artifact_id), context={})

    assert response["statusCode"] == 500
    body = json.loads(response["body"])
    assert "deletion incomplete" in body["error"].lower()
    # Should report blob, metadata, name_index, and rating failures
    assert "blob" in body["error"]
    assert "metadata" in body["error"]
    assert "name_index" in body["error"]
    assert "rating" in body["error"]


def test_delete_invalid_artifact_type() -> None:
    """Test deletion with invalid artifact type returns 400."""
    response = handler.lambda_handler(
        _event("invalid", "test123"), context={}
    )

    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert "invalid" in body["error"].lower()


def test_delete_missing_artifact_id() -> None:
    """Test deletion without artifact ID returns 400."""
    token = auth.issue_token("tester", is_admin=True)
    event = {
        "pathParameters": {"artifact_type": "model"},
        "headers": {"X-Authorization": token},
    }

    response = handler.lambda_handler(event, context={})

    assert response["statusCode"] == 400


class _FakeMetadataStore:
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


class _FakeNameIndex:
    def __init__(self) -> None:
        self.should_fail = False

    def delete(self, entry: Any) -> None:
        if self.should_fail:
            raise Exception("Name index deletion failed")
