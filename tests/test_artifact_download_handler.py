"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for GET /artifact/{artifact_type}/{id}/download handler.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from backend.src.handlers.artifact_download import app as handler
from src.storage.blob_store import (BlobNotFoundError, BlobStoreError,
                                    DownloadLink, StoredArtifact)
from src.utils import auth


class _FakeStore:
    """
    _FakeStore: Class description.
    """

    def __init__(self, *, should_fail: str | None = None) -> None:
        """
        __init__: Function description.
        :param should_fail:
        :returns:
        """

        self.should_fail = should_fail

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

        raise NotImplementedError

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

        raise NotImplementedError

    def generate_download_url(
        self, artifact_id: str, *, expires_in: int = 900
    ) -> DownloadLink:
        """
        generate_download_url: Function description.
        :param artifact_id:
        :param expires_in:
        :returns:
        """

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
    *,
    query: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """
    _event: Function description.
    :param artifact_id:
    :param query:
    :returns:
    """

    token = auth.issue_token("tester", is_admin=True)
    return {
        "pathParameters": {
            "id": artifact_id,
        },
        "headers": {"X-Authorization": token},
        "queryStringParameters": query,
    }


def test_download_success() -> None:
    """
    test_download_success: Function description.
    :param:
    :returns:
    """

    handler._BLOB_STORE = _FakeStore()  # type: ignore[attr-defined]
    response = handler.lambda_handler(_event(), context={})
    assert response["statusCode"] == 302
    assert response["headers"]["Location"].startswith("https://downloads/")


def test_download_invalid_id() -> None:
    """
    test_download_invalid_id: Function description.
    :param:
    :returns:
    """

    handler._BLOB_STORE = _FakeStore()  # type: ignore[attr-defined]
    response = handler.lambda_handler(
        _event(artifact_id="not valid!"), context={}
    )
    assert response["statusCode"] == 400


def test_download_not_found() -> None:
    """
    test_download_not_found: Function description.
    :param:
    :returns:
    """

    handler._BLOB_STORE = _FakeStore(
        should_fail="notfound"
    )  # type: ignore[attr-defined]
    response = handler.lambda_handler(_event(), context={})
    assert response["statusCode"] == 404


def test_download_blob_failure() -> None:
    """
    test_download_blob_failure: Function description.
    :param:
    :returns:
    """

    handler._BLOB_STORE = _FakeStore(
        should_fail="error"
    )  # type: ignore[attr-defined]
    response = handler.lambda_handler(_event(), context={})
    assert response["statusCode"] == 502


def test_download_json_format() -> None:
    """
    test_download_json_format: Function description.
    :param:
    :returns:
    """

    handler._BLOB_STORE = _FakeStore()  # type: ignore[attr-defined]
    response = handler.lambda_handler(
        _event(query={"format": "json"}), context={}
    )
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["artifact_id"] == "1234abcd"
