"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for the PUT /artifacts/{artifact_type}/{id} handler.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, cast

import pytest

from backend.src.handlers.artifact_update import app as handler
from src.models import Artifact
from src.models.artifacts import ArtifactData, ArtifactMetadata, ArtifactType
from src.storage.artifact_ingest import ArtifactBundle
from src.storage.blob_store import StoredArtifact
from src.storage.errors import ArtifactNotFound, ValidationError
from src.storage.metadata_store import ArtifactMetadataStore
from src.utils import auth


@pytest.fixture(autouse=True)
def _reset_handler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    _reset_handler: Function description.
    :param monkeypatch:
    :param tmp_path:
    :returns:
    """

    store = _FakeMetadataStore()
    handler._METADATA_STORE = cast(ArtifactMetadataStore, store)
    handler._BLOB_STORE = _FakeBlobStore()

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "file.txt").write_text("new content", encoding="utf-8")

    def _fake_prepare(url: str) -> ArtifactBundle:
        """
        _fake_prepare: Function description.
        :param url:
        :returns:
        """

        return ArtifactBundle(
            kind="directory",
            path=bundle_dir,
            cleanup_root=bundle_dir,
            content_type="application/gzip",
        )

    monkeypatch.setattr(handler, "prepare_artifact_bundle", _fake_prepare)

    store.save(
        Artifact(
            metadata=ArtifactMetadata(
                name="whisper-tiny",
                id="artifact123",
                type=ArtifactType.MODEL,
            ),
            data=ArtifactData(url="https://example.com/old"),
        )
    )


def _event(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    _event: Function description.
    :param body:
    :returns:
    """

    token = auth.issue_token("tester", is_admin=True)
    return {
        "pathParameters": {"artifact_type": "model", "id": "artifact123"},
        "headers": {"X-Authorization": token},
        "body": json.dumps(body),
    }


def test_update_artifact_success() -> None:
    """
    test_update_artifact_success: Function description.
    :param:
    :returns:
    """

    payload = {
        "metadata": {
            "name": "whisper-tiny",
            "id": "artifact123",
            "type": "model",
        },
        "data": {"url": "https://example.com/new-model"},
    }

    response = handler.lambda_handler(_event(payload), context={})

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["data"]["url"] == "https://example.com/new-model"


def test_update_rejects_name_mismatch() -> None:
    """
    test_update_rejects_name_mismatch: Function description.
    :param:
    :returns:
    """

    payload = {
        "metadata": {
            "name": "different",
            "id": "artifact123",
            "type": "model",
        },
        "data": {"url": "https://example.com/new-model"},
    }

    response = handler.lambda_handler(_event(payload), context={})

    assert response["statusCode"] == 400


def test_update_missing_artifact_returns_404() -> None:
    """
    test_update_missing_artifact_returns_404: Function description.
    :param:
    :returns:
    """

    handler._METADATA_STORE = cast(
        ArtifactMetadataStore, _FakeMetadataStore()
    )
    payload = {
        "metadata": {
            "name": "whisper-tiny",
            "id": "artifact123",
            "type": "model",
        },
        "data": {"url": "https://example.com/new-model"},
    }

    response = handler.lambda_handler(_event(payload), context={})

    assert response["statusCode"] == 404


def test_update_metadata_id_mismatch() -> None:
    """
    test_update_metadata_id_mismatch: Function description.
    :param:
    :returns:
    """

    payload = {
        "metadata": {
            "name": "whisper-tiny",
            "id": "other",
            "type": "model",
        },
        "data": {"url": "https://example.com/new-model"},
    }

    response = handler.lambda_handler(_event(payload), context={})

    assert response["statusCode"] == 400


class _FakeBlobStore:
    """
    _FakeBlobStore: Class description.
    """

    def __init__(self) -> None:
        """
        __init__: Function description.
        :param:
        :returns:
        """

        self.saved: list[tuple[str, Path]] = []

    def store_file(
        self,
        artifact_id: str,
        file_path: Path,
        *,
        content_type: str | None = None,
    ) -> StoredArtifact:
        """
        store_file: Function description.
        :param artifact_id:
        :param file_path:
        :param content_type:
        :returns:
        """

        self.saved.append((artifact_id, file_path))
        return StoredArtifact(
            artifact_id=artifact_id,
            uri=str(file_path),
            bytes_written=1,
            content_type=content_type,
        )

    def store_directory(
        self,
        artifact_id: str,
        directory: Path,
        *,
        content_type: str | None = None,
    ) -> StoredArtifact:
        """
        store_directory: Function description.
        :param artifact_id:
        :param directory:
        :param content_type:
        :returns:
        """

        return self.store_file(
            artifact_id,
            directory,
            content_type=content_type,
        )

    def generate_download_url(
        self,
        artifact_id: str,
        *,
        expires_in: int = 900,
    ):
        """
        generate_download_url: Function description.
        :param artifact_id:
        :param expires_in:
        :returns:
        """

        raise NotImplementedError


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
