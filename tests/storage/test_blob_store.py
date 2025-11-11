"""Tests for artifact blob stores."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.storage.blob_store import (BlobNotFoundError, DownloadLink,
                                    LocalArtifactBlobStore, StoredArtifact)


def test_local_blob_store_writes_file(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"binary-data")
    store = LocalArtifactBlobStore(tmp_path / "artifacts")

    result = store.store_file("artifact123", payload)
    assert isinstance(result, StoredArtifact)
    destination = tmp_path / "artifacts" / "artifact123"
    assert destination.read_bytes() == payload.read_bytes()
    assert result.bytes_written == payload.stat().st_size
    link = store.generate_download_url("artifact123")
    assert isinstance(link, DownloadLink)
    assert link.url.startswith("file://")


def test_local_blob_store_missing_download(tmp_path: Path) -> None:
    store = LocalArtifactBlobStore(tmp_path)
    with pytest.raises(BlobNotFoundError):
        store.generate_download_url("missing")
