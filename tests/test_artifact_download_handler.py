"""Tests for GET /artifact/{artifact_type}/{id}/download handler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from backend.src.handlers.artifact_download import app as handler
from src.storage.blob_store import (BlobNotFoundError, BlobStoreError,
                                    DownloadLink, StoredArtifact)


class _FakeStore:
    def __init__(self, *, should_fail: str | None = None) -> None:
        self.should_fail = should_fail

    def store_file(
        self,
        artifact_id: str,
        file_path: Path,
        *,
        content_type: str | None = None,
    ) -> StoredArtifact:
        raise NotImplementedError

    def generate_download_url(
        self, artifact_id: str, *, expires_in: int = 900
    ) -> DownloadLink:
        if self.should_fail == "notfound":
            raise BlobNotFoundError("missing")
        if self.should_fail == "error":
            raise BlobStoreError("boom")
        return DownloadLink(
            artifact_id=artifact_id,
            url=f"https://downloads/{artifact_id}",
            expires_in=expires_in,
        )


def _event(
    artifact_id: str = "1234abcd",
) -> Dict[str, Any]:
    return {
        "pathParameters": {
            "id": artifact_id,
        },
        "headers": {"X-Authorization": "token"},
    }


def test_download_success() -> None:
    handler._BLOB_STORE = _FakeStore()  # type: ignore[attr-defined]
    response = handler.lambda_handler(_event(), context={})
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["download_url"].startswith("https://downloads/")
    assert body["artifact_id"] == "1234abcd"


def test_download_invalid_id() -> None:
    handler._BLOB_STORE = _FakeStore()  # type: ignore[attr-defined]
    response = handler.lambda_handler(
        _event(artifact_id="not valid!"), context={}
    )
    assert response["statusCode"] == 400


def test_download_not_found() -> None:
    handler._BLOB_STORE = _FakeStore(
        should_fail="notfound"
    )  # type: ignore[attr-defined]
    response = handler.lambda_handler(_event(), context={})
    assert response["statusCode"] == 404


def test_download_blob_failure() -> None:
    handler._BLOB_STORE = _FakeStore(
        should_fail="error"
    )  # type: ignore[attr-defined]
    response = handler.lambda_handler(_event(), context={})
    assert response["statusCode"] == 502
