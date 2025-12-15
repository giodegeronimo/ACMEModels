"""

ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

"""
from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from backend.src.handlers.artifact_create import app as create_handler
from backend.src.handlers.artifact_download import app as download_handler
from backend.src.handlers.artifact_update import app as update_handler
from src.models.artifacts import (Artifact, ArtifactData, ArtifactMetadata,
                                  ArtifactType)
from src.storage.blob_store import BlobStoreUnavailableError, DownloadLink
from src.storage.metadata_store import ArtifactMetadataStore
from src.utils import auth


def _auth_headers() -> dict[str, str]:
    """
    _auth_headers: Function description.
    :param:
    :returns:
    """

    token = auth.issue_token("tester", is_admin=True)
    return {"X-Authorization": token}


def test_create_returns_503_when_blob_store_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_create_returns_503_when_blob_store_unavailable: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("AWS_SAM_LOCAL", "1")
    monkeypatch.setenv("ACME_DISABLE_ASYNC", "1")
    monkeypatch.setattr(create_handler, "_can_process_synchronously", lambda url: True)

    class ExplodingBlobStore:
        """
        ExplodingBlobStore: Class description.
        """

        def store_file(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
            """
            store_file: Function description.
            :param *args:
            :param **kwargs:
            :returns:
            """

            raise BlobStoreUnavailableError("S3 down")

        def store_directory(self, *args: Any, **kwargs: Any):  # pragma: no cover
            """
            store_directory: Function description.
            :param *args:
            :param **kwargs:
            :returns:
            """

            raise BlobStoreUnavailableError("S3 down")

    class SpyMetadataStore(ArtifactMetadataStore):
        """
        SpyMetadataStore: Class description.
        """

        def __init__(self) -> None:
            """
            __init__: Function description.
            :param:
            :returns:
            """

            self.saved = 0

        def save(self, artifact: Artifact, *, overwrite: bool = False) -> None:
            """
            save: Function description.
            :param artifact:
            :param overwrite:
            :returns:
            """

            self.saved += 1

        def load(self, artifact_id: str) -> Artifact:  # pragma: no cover
            """
            load: Function description.
            :param artifact_id:
            :returns:
            """

            raise RuntimeError("not used")

    spy_meta = SpyMetadataStore()
    monkeypatch.setattr(create_handler, "_BLOB_STORE", ExplodingBlobStore())
    monkeypatch.setattr(create_handler, "_METADATA_STORE", spy_meta)
    monkeypatch.setattr(create_handler, "_NAME_INDEX", type("NI", (), {"save": lambda *_a, **_k: None})())
    monkeypatch.setattr(create_handler, "_ensure_stub_rating_exists", lambda *_a, **_k: None)

    # Avoid network/download by returning a dummy bundle.
    import tempfile
    from pathlib import Path

    from src.storage.artifact_ingest import ArtifactBundle

    dummy = Path(tempfile.mkstemp(prefix="bundle-", suffix=".bin")[1])
    dummy.write_bytes(b"x")
    monkeypatch.setattr(
        create_handler,
        "prepare_artifact_bundle",
        lambda url: ArtifactBundle(kind="file", path=dummy, cleanup_root=dummy),
    )
    monkeypatch.setattr(create_handler, "_compute_and_store_rating_if_needed", lambda *_a, **_k: None)
    monkeypatch.setattr(create_handler, "_extract_and_store_lineage", lambda *_a, **_k: None)

    event = {
        "pathParameters": {"artifact_type": "model"},
        "headers": _auth_headers(),
        "body": json.dumps({"url": "https://huggingface.co/org/model"}),
        "isBase64Encoded": False,
    }
    context = type("Ctx", (), {"invoked_function_arn": "arn:aws:lambda:test"})

    response = create_handler.lambda_handler(event, context)
    assert response["statusCode"] == 503
    assert "temporarily unavailable" in response["body"]
    assert spy_meta.saved == 0


def test_download_returns_503_when_blob_store_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_download_returns_503_when_blob_store_unavailable: Function description.
    :param monkeypatch:
    :returns:
    """

    class ExplodingBlobStore:
        """
        ExplodingBlobStore: Class description.
        """

        def generate_download_url(self, *args: Any, **kwargs: Any) -> DownloadLink:  # type: ignore[no-untyped-def]
            """
            generate_download_url: Function description.
            :param *args:
            :param **kwargs:
            :returns:
            """

            raise BlobStoreUnavailableError("S3 down")

    monkeypatch.setattr(download_handler, "_BLOB_STORE", ExplodingBlobStore())

    event = {
        "pathParameters": {"id": "abc123"},
        "headers": _auth_headers(),
        "queryStringParameters": {"format": "json"},
    }
    response = download_handler.lambda_handler(event, None)
    assert response["statusCode"] == 503


def test_update_returns_503_when_blob_store_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_update_returns_503_when_blob_store_unavailable: Function description.
    :param monkeypatch:
    :returns:
    """

    class ExplodingBlobStore:
        """
        ExplodingBlobStore: Class description.
        """

        def store_file(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
            """
            store_file: Function description.
            :param *args:
            :param **kwargs:
            :returns:
            """

            raise BlobStoreUnavailableError("S3 down")

        def store_directory(self, *args: Any, **kwargs: Any):  # pragma: no cover
            """
            store_directory: Function description.
            :param *args:
            :param **kwargs:
            :returns:
            """

            raise BlobStoreUnavailableError("S3 down")

    class SpyMetadataStore(ArtifactMetadataStore):
        """
        SpyMetadataStore: Class description.
        """

        def __init__(self, artifact: Artifact) -> None:
            """
            __init__: Function description.
            :param artifact:
            :returns:
            """

            self._artifact = artifact
            self.saved = 0

        def save(self, artifact: Artifact, *, overwrite: bool = False) -> None:
            """
            save: Function description.
            :param artifact:
            :param overwrite:
            :returns:
            """

            self.saved += 1

        def load(self, artifact_id: str) -> Artifact:
            """
            load: Function description.
            :param artifact_id:
            :returns:
            """

            return self._artifact

    existing = Artifact(
        metadata=ArtifactMetadata(name="demo", id="abc123", type=ArtifactType.MODEL),
        data=ArtifactData(url="https://huggingface.co/org/model"),
    )
    spy_meta = SpyMetadataStore(existing)
    monkeypatch.setattr(update_handler, "_BLOB_STORE", ExplodingBlobStore())
    monkeypatch.setattr(update_handler, "_METADATA_STORE", spy_meta)
    monkeypatch.setenv("AWS_SAM_LOCAL", "1")

    import tempfile
    from pathlib import Path

    from src.storage.artifact_ingest import ArtifactBundle

    dummy = Path(tempfile.mkdtemp(prefix="bundle-"))
    (dummy / "file.bin").write_bytes(b"x")
    monkeypatch.setattr(
        update_handler,
        "prepare_artifact_bundle",
        lambda url: ArtifactBundle(
            kind="file",
            path=dummy / "file.bin",
            cleanup_root=dummy / "file.bin",
            content_type="application/octet-stream",
        ),
    )

    event: Dict[str, Any] = {
        "pathParameters": {"artifact_type": "model", "id": "abc123"},
        "headers": _auth_headers(),
        "body": json.dumps(
            {
                "metadata": {"name": "demo", "id": "abc123", "type": "model"},
                "data": {"url": "https://huggingface.co/org/model2"},
            }
        ),
        "isBase64Encoded": False,
    }

    response = update_handler.lambda_handler(event, None)
    assert response["statusCode"] == 503
    assert spy_meta.saved == 0
