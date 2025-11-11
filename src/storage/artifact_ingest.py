"""Helpers for preparing artifact bundles prior to storage."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

import requests
from huggingface_hub import snapshot_download


class ArtifactDownloadError(RuntimeError):
    """Raised when an artifact cannot be prepared for ingest."""


def prepare_artifact_bundle(source_url: str) -> Tuple[Path, str | None]:
    """
    Download and package the artifact referenced by ``source_url``.

    Returns a tuple of (path_to_file, content_type).
    """

    parsed = urlparse(source_url)
    if parsed.netloc.endswith("huggingface.co"):
        return _download_huggingface_repo(parsed)
    return _download_generic_file(source_url)


def _download_generic_file(source_url: str) -> Tuple[Path, str | None]:
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
    return path, content_type


def _download_huggingface_repo(
    parsed_url,
) -> Tuple[Path, str | None]:
    os.environ.setdefault("HF_HOME", "/tmp/hf-cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf-cache")
    repo_id = _parse_hf_repo_id(parsed_url.path)
    if not repo_id:
        raise ArtifactDownloadError(
            "Unable to determine Hugging Face repository from URL"
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="hf_repo_"))
    try:
        snapshot_path = snapshot_download(  # type: ignore[arg-type]
            repo_id=repo_id,
            repo_type="model",
            local_dir=temp_dir / "repo",
            cache_dir=temp_dir / "cache",
            local_dir_use_symlinks=False,
        )
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ArtifactDownloadError(
            f"Failed to download Hugging Face repository: {exc}"
        ) from exc

    archive_base = temp_dir / "artifact"
    archive_path = Path(
        shutil.make_archive(str(archive_base), "zip", snapshot_path)
    )
    return archive_path, "application/zip"


def _parse_hf_repo_id(path: str) -> str | None:
    segments = [segment for segment in path.split("/") if segment]
    if len(segments) < 2:
        return None
    # Path can be /org/model/blob/main/...; we only care about first two.
    return "/".join(segments[:2])
