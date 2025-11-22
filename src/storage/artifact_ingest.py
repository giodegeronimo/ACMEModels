"""Helpers for preparing artifact bundles prior to storage."""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import requests
from huggingface_hub import HfApi, hf_hub_download


class ArtifactDownloadError(RuntimeError):
    """Raised when an artifact cannot be prepared for ingest."""


@dataclass(frozen=True)
class ArtifactBundle:
    """Represents downloaded artifact contents awaiting storage."""

    kind: Literal["file", "directory"]
    path: Path
    cleanup_root: Path
    content_type: str | None = None
    readme_excerpt: str | None = None


_README_LIMIT_ENV = os.environ.get("ARTIFACT_README_MAX_BYTES", "0")
try:
    _README_LIMIT_VALUE = int(_README_LIMIT_ENV)
except ValueError:
    _README_LIMIT_VALUE = 0
README_CAPTURE_LIMIT = _README_LIMIT_VALUE if _README_LIMIT_VALUE > 0 else None


def prepare_artifact_bundle(source_url: str) -> ArtifactBundle:
    """Download and stage the artifact referenced by ``source_url``."""

    parsed = urlparse(source_url)
    if parsed.netloc.endswith("huggingface.co"):
        if "/resolve/" in parsed.path:
            return _download_generic_file(source_url)
        repo_info = _parse_hf_repo_info(parsed.path)
        if not repo_info:
            raise ArtifactDownloadError(
                "Unable to determine Hugging Face repository from URL"
            )
        return _download_huggingface_repo(parsed, repo_info)
    if parsed.netloc.endswith("github.com"):
        repo = _parse_github_repo(parsed.path)
        if repo:
            return _download_github_repo(parsed, repo)
    return _download_generic_file(source_url)


def _stream_to_temp_file(
    source_url: str,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[Path, str | None]:
    try:
        response = requests.get(
            source_url,
            stream=True,
            timeout=30,
            headers=headers,
        )
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

    return path, response.headers.get("Content-Type")


def _is_readme_filename(name: str) -> bool:
    base = Path(name).name.lower()
    return base in {"readme", "readme.md", "readme.txt", "readme.rst"}


def _read_text_file(
    path: Path,
    limit: int | None = README_CAPTURE_LIMIT,
) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="latin-1")
        except Exception:
            return None
    text = text.strip()
    if not text:
        return None
    if limit is not None and len(text) > limit:
        return text[:limit]
    return text


def _extract_readme_from_directory(root: Path) -> str | None:
    candidates = sorted(root.rglob("*"))
    for candidate in candidates:
        if candidate.is_file() and _is_readme_filename(candidate.name):
            excerpt = _read_text_file(candidate)
            if excerpt:
                return excerpt
    return None


def _extract_readme_from_zip(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                if _is_readme_filename(info.filename):
                    with archive.open(info) as handle:
                        data = handle.read()
                        text = data.decode("utf-8", errors="ignore").strip()
                        if text:
                            if (
                                README_CAPTURE_LIMIT is not None
                                and len(text) > README_CAPTURE_LIMIT
                            ):
                                return text[:README_CAPTURE_LIMIT]
                            return text
    except Exception:
        return None
    return None


def _download_generic_file(source_url: str) -> ArtifactBundle:
    path, content_type = _stream_to_temp_file(source_url)
    return ArtifactBundle(
        kind="file",
        path=path,
        cleanup_root=path,
        content_type=content_type,
    )


def _download_huggingface_repo(
    parsed_url,
    repo_info: tuple[str, str],
) -> ArtifactBundle:
    os.environ.setdefault("HF_HOME", "/tmp/hf-cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf-cache")
    repo_type, repo_id = repo_info

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    except Exception as exc:  # noqa: BLE001
        raise ArtifactDownloadError(
            f"Failed to enumerate Hugging Face repository: {exc}"
        ) from exc
    if not files:
        raise ArtifactDownloadError("Hugging Face repository is empty")

    temp_dir = Path(tempfile.mkdtemp(prefix="hf_repo_"))
    downloads_dir = temp_dir / "downloads"
    cache_dir = temp_dir / "cache"
    archive_path = temp_dir / "repo.tar.gz"
    readme_excerpt: str | None = None

    try:
        with tarfile.open(archive_path, "w:gz") as archive:
            for file_path in files:
                local_file = Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=file_path,
                        repo_type=repo_type,
                        cache_dir=cache_dir,
                        local_dir=downloads_dir,
                    )
                )
                archive.add(local_file, arcname=file_path)
        readme_excerpt = _extract_readme_from_directory(downloads_dir)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ArtifactDownloadError(
            f"Failed to download Hugging Face repository: {exc}"
        ) from exc
    finally:
        shutil.rmtree(downloads_dir, ignore_errors=True)

    return ArtifactBundle(
        kind="file",
        path=archive_path,
        cleanup_root=temp_dir,
        content_type="application/gzip",
        readme_excerpt=readme_excerpt,
    )


def _parse_hf_repo_info(path: str) -> tuple[str, str] | None:
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return None
    repo_type = "model"
    prefix_map = {"datasets": "dataset", "spaces": "space", "models": "model"}
    if segments[0] in prefix_map:
        repo_type = prefix_map[segments[0]]
        segments = segments[1:]
    if not segments:
        return None
    if "resolve" in segments:
        resolve_index = segments.index("resolve")
        segments = segments[:resolve_index]
    if not segments:
        return None
    if len(segments) == 1:
        repo_id = segments[0]
    else:
        repo_id = "/".join(segments[:2])
    return repo_type, repo_id


def _parse_github_repo(path: str) -> tuple[str, str] | None:
    segments = [segment for segment in path.split("/") if segment]
    if len(segments) < 2:
        return None
    owner, repo = segments[:2]
    if owner in {"repos", "users"}:  # handle API-style URLs
        if len(segments) >= 4:
            owner, repo = segments[1:3]
        else:
            return None
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


def _download_github_repo(parsed_url, repo: tuple[str, str]) -> ArtifactBundle:
    owner, name = repo
    branch = _resolve_github_default_branch(owner, name)
    candidate_branches = [branch] if branch else []
    candidate_branches.extend(["main", "master"])
    candidate_branches = list(dict.fromkeys(candidate_branches))

    headers = {}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_error: Exception | None = None
    for candidate in candidate_branches:
        if candidate is None:
            continue
        archive_url = (
            "https://codeload.github.com/"
            f"{owner}/{name}/zip/refs/heads/{candidate}"
        )
        try:
            path, content_type = _stream_to_temp_file(
                archive_url,
                headers=headers,
            )
            readme_excerpt = _extract_readme_from_zip(path)
            return ArtifactBundle(
                kind="file",
                path=path,
                cleanup_root=path,
                content_type=content_type or "application/zip",
                readme_excerpt=readme_excerpt,
            )
        except ArtifactDownloadError as exc:
            last_error = exc
            continue
    raise ArtifactDownloadError(
        "Failed to download GitHub repository archive "
        f"for {owner}/{name}: {last_error}"
    )


def _resolve_github_default_branch(owner: str, repo: str) -> str | None:
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            branch = data.get("default_branch")
            if isinstance(branch, str) and branch:
                return branch
    except Exception:
        return None
    return None
