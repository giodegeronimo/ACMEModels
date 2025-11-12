"""Helpers for preparing artifact bundles prior to storage."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import requests
from huggingface_hub import snapshot_download


class ArtifactDownloadError(RuntimeError):
    """Raised when an artifact cannot be prepared for ingest."""


@dataclass(frozen=True)
class ArtifactBundle:
    """Represents downloaded artifact contents awaiting storage."""

    kind: Literal["file", "directory"]
    path: Path
    cleanup_root: Path
    content_type: str | None = None


def prepare_artifact_bundle(source_url: str) -> ArtifactBundle:
    """Download and stage the artifact referenced by ``source_url``."""

    parsed = urlparse(source_url)
    if parsed.netloc.endswith("huggingface.co"):
        return _download_huggingface_repo(parsed)
    return _download_generic_file(source_url)


def _download_generic_file(source_url: str) -> ArtifactBundle:
    try:
        response = requests.get(source_url, stream=True, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ArtifactDownloadError(
            f"Failed to download artifact from '{source_url}': {exc}"
        ) from exc

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    path = Path(temp_file.name)
    try:
        with temp_file as handle:
            for chunk in response.iter_content(8192):
                if chunk:
                    handle.write(chunk)
    except Exception as exc:  # noqa: BLE001
        path.unlink(missing_ok=True)
        raise ArtifactDownloadError(
            f"Failed to stream artifact: {exc}"
        ) from exc

    content_type = response.headers.get("Content-Type")
    return ArtifactBundle(
        kind="file",
        path=path,
        cleanup_root=path,
        content_type=content_type,
    )


def _download_huggingface_repo(parsed_url) -> ArtifactBundle:
    os.environ.setdefault("HF_HOME", "/tmp/hf-cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf-cache")
    repo_id = _parse_hf_repo_id(parsed_url.path)
    if not repo_id:
        raise ArtifactDownloadError(
            "Unable to determine Hugging Face repository from URL"
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="hf_repo_"))
    try:
        snapshot_path = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=temp_dir / "repo",
                cache_dir=temp_dir / "cache",
            )
        )
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ArtifactDownloadError(
            f"Failed to download Hugging Face repository: {exc}"
        ) from exc

    return ArtifactBundle(
        kind="directory",
        path=snapshot_path,
        cleanup_root=temp_dir,
        content_type="application/gzip",
    )


def _parse_hf_repo_id(path: str) -> str | None:
    segments = [segment for segment in path.split("/") if segment]
    if len(segments) < 2:
        return None
    return "/".join(segments[:2])
