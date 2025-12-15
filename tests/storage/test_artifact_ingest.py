"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Unit tests for artifact ingest helpers.
"""

from __future__ import annotations

import importlib
import io
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterator, Tuple

import pytest

from src.storage import artifact_ingest as ingest


def test_extract_readme_from_directory(tmp_path: Path) -> None:
    """
    test_extract_readme_from_directory: Function description.
    :param tmp_path:
    :returns:
    """

    readme = tmp_path / "README.md"
    readme.write_text("Hello directory README", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "other.txt").write_text("ignored", encoding="utf-8")

    excerpt = ingest._extract_readme_from_directory(tmp_path)

    assert excerpt == "Hello directory README"


def test_extract_readme_from_directory_returns_none_when_missing(
    tmp_path: Path,
) -> None:
    """
    test_extract_readme_from_directory_returns_none_when_missing: Function description.
    :param tmp_path:
    :returns:
    """

    assert ingest._extract_readme_from_directory(tmp_path) is None


def test_extract_readme_from_zip(tmp_path: Path) -> None:
    """
    test_extract_readme_from_zip: Function description.
    :param tmp_path:
    :returns:
    """

    archive = tmp_path / "repo.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("repo/README.txt", "Zip README content")
        zf.writestr("repo/other.md", "ignored")

    excerpt = ingest._extract_readme_from_zip(archive)

    assert excerpt == "Zip README content"


def test_extract_readme_from_zip_skips_directories_and_respects_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_extract_readme_from_zip_skips_directories_and_respects_limit: Function description.
    :param tmp_path:
    :param monkeypatch:
    :returns:
    """

    archive = tmp_path / "repo.zip"
    long_text = "x" * 500
    monkeypatch.setattr(ingest, "README_CAPTURE_LIMIT", 10)
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("repo/", "")
        zf.writestr("repo/README.md", long_text)

    assert ingest._extract_readme_from_zip(archive) == "x" * 10


def test_extract_readme_from_zip_returns_none_when_no_readme(tmp_path: Path) -> None:
    """
    test_extract_readme_from_zip_returns_none_when_no_readme: Function description.
    :param tmp_path:
    :returns:
    """

    archive = tmp_path / "repo.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("repo/file.txt", "content")

    assert ingest._extract_readme_from_zip(archive) is None


def test_extract_readme_directory_preserves_full_content(
    tmp_path: Path,
) -> None:
    """
    test_extract_readme_directory_preserves_full_content: Function description.
    :param tmp_path:
    :returns:
    """

    readme = tmp_path / "README"
    readme.write_text("A" * 5000 + "tail", encoding="utf-8")

    excerpt = ingest._extract_readme_from_directory(tmp_path)

    assert excerpt is not None
    assert excerpt.endswith("tail")


def test_extract_readme_zip_preserves_full_content(
    tmp_path: Path,
) -> None:
    """
    test_extract_readme_zip_preserves_full_content: Function description.
    :param tmp_path:
    :returns:
    """

    archive = tmp_path / "repo.zip"
    content = "B" * 7000 + "end"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("repo/README.md", content)

    excerpt = ingest._extract_readme_from_zip(archive)

    assert excerpt is not None
    assert excerpt.endswith("end")


def test_is_readme_filename_matches_common_variants() -> None:
    """
    test_is_readme_filename_matches_common_variants: Function description.
    :param:
    :returns:
    """

    assert ingest._is_readme_filename("README")
    assert ingest._is_readme_filename("readme.md")
    assert ingest._is_readme_filename("ReadMe.TXT")
    assert not ingest._is_readme_filename("not_readme.md")


def test_read_text_file_trims_and_respects_limit(tmp_path: Path) -> None:
    """
    test_read_text_file_trims_and_respects_limit: Function description.
    :param tmp_path:
    :returns:
    """

    path = tmp_path / "README.md"
    path.write_text("  hello world  ", encoding="utf-8")

    assert ingest._read_text_file(path) == "hello world"
    assert ingest._read_text_file(path, limit=5) == "hello"


def test_read_text_file_returns_none_for_empty_content(tmp_path: Path) -> None:
    """
    test_read_text_file_returns_none_for_empty_content: Function description.
    :param tmp_path:
    :returns:
    """

    path = tmp_path / "README.md"
    path.write_text("   \n", encoding="utf-8")
    assert ingest._read_text_file(path) is None


def test_read_text_file_returns_none_when_decode_fails_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_read_text_file_returns_none_when_decode_fails_twice: Function description.
    :param tmp_path:
    :param monkeypatch:
    :returns:
    """

    path = tmp_path / "README.md"
    path.write_bytes(b"\xff")

    original_read_text = Path.read_text

    def fake_read_text(self: Path, *, encoding: str = "utf-8", **kwargs: Any) -> str:
        """
        fake_read_text: Function description.
        :param encoding:
        :param **kwargs:
        :returns:
        """

        if encoding == "utf-8":
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
        raise RuntimeError("no decoder")

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    try:
        assert ingest._read_text_file(path) is None
    finally:
        monkeypatch.setattr(Path, "read_text", original_read_text)


def test_read_text_file_falls_back_to_latin1(tmp_path: Path) -> None:
    """
    test_read_text_file_falls_back_to_latin1: Function description.
    :param tmp_path:
    :returns:
    """

    readme = tmp_path / "README.md"
    readme.write_bytes(b"\xff\n")

    assert ingest._read_text_file(readme) == "ÿ"


def test_parse_hf_repo_info_handles_prefixes_and_resolve() -> None:
    """
    test_parse_hf_repo_info_handles_prefixes_and_resolve: Function description.
    :param:
    :returns:
    """

    assert ingest._parse_hf_repo_info("/org/model") == ("model", "org/model")
    assert ingest._parse_hf_repo_info("/models/org/model") == (
        "model",
        "org/model",
    )
    assert ingest._parse_hf_repo_info("/datasets/org/ds") == (
        "dataset",
        "org/ds",
    )
    assert ingest._parse_hf_repo_info("/org/model/resolve/main/file.bin") == (
        "model",
        "org/model",
    )
    assert ingest._parse_hf_repo_info("/") is None
    assert ingest._parse_hf_repo_info("/datasets") is None
    assert ingest._parse_hf_repo_info("/resolve/main/file") is None


def test_parse_github_repo_strips_git_suffix_and_api_style() -> None:
    """
    test_parse_github_repo_strips_git_suffix_and_api_style: Function description.
    :param:
    :returns:
    """

    assert ingest._parse_github_repo("/org/repo.git") == ("org", "repo")
    assert ingest._parse_github_repo("/repos/org/repo") == ("org", "repo")
    assert ingest._parse_github_repo("/users/org/repo") == ("org", "repo")
    assert ingest._parse_github_repo("/repos/org") is None
    assert ingest._parse_github_repo("/org") is None


class _FakeResponse:
    """
    _FakeResponse: Class description.
    """

    def __init__(self, *, content: bytes, content_type: str = "text/plain") -> None:
        """
        __init__: Function description.
        :param content:
        :param content_type:
        :returns:
        """

        self._content = content
        self.headers = {"Content-Type": content_type}
        self.status_code = 200

    def raise_for_status(self) -> None:
        """
        raise_for_status: Function description.
        :param:
        :returns:
        """

        return None

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        """
        iter_content: Function description.
        :param chunk_size:
        :returns:
        """

        buffer = io.BytesIO(self._content)
        while True:
            chunk = buffer.read(chunk_size)
            if not chunk:
                break
            yield chunk


def test_stream_to_temp_file_downloads_and_returns_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_stream_to_temp_file_downloads_and_returns_content_type: Function description.
    :param monkeypatch:
    :returns:
    """

    response = _FakeResponse(content=b"payload", content_type="application/bin")

    def fake_get(*_: Any, **__: Any) -> _FakeResponse:
        """
        fake_get: Function description.
        :param *_:
        :param **__:
        :returns:
        """

        return response

    monkeypatch.setattr(ingest.requests, "get", fake_get)
    path, content_type = ingest._stream_to_temp_file("https://example.com/x")
    try:
        assert path.read_bytes() == b"payload"
        assert content_type == "application/bin"
    finally:
        path.unlink(missing_ok=True)


def test_stream_to_temp_file_wraps_request_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_stream_to_temp_file_wraps_request_errors: Function description.
    :param monkeypatch:
    :returns:
    """

    def fake_get(*_: Any, **__: Any) -> Any:
        """
        fake_get: Function description.
        :param *_:
        :param **__:
        :returns:
        """

        raise ingest.requests.RequestException("boom")

    monkeypatch.setattr(ingest.requests, "get", fake_get)

    with pytest.raises(ingest.ArtifactDownloadError, match="Failed to download"):
        ingest._stream_to_temp_file("https://example.com/file.bin")


def test_stream_to_temp_file_cleans_up_on_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    test_stream_to_temp_file_cleans_up_on_write_failure: Function description.
    :param monkeypatch:
    :param tmp_path:
    :returns:
    """

    response = _FakeResponse(content=b"payload")

    def fake_get(*_: Any, **__: Any) -> _FakeResponse:
        """
        fake_get: Function description.
        :param *_:
        :param **__:
        :returns:
        """

        return response

    monkeypatch.setattr(ingest.requests, "get", fake_get)

    class _Temp:
        """
        _Temp: Class description.
        """

        def __init__(self) -> None:
            """
            __init__: Function description.
            :param:
            :returns:
            """

            self.name = str(tmp_path / "temp.bin")

        def __enter__(self) -> "_Temp":
            """
            __enter__: Function description.
            :param:
            :returns:
            """

            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            """
            __exit__: Function description.
            :param exc_type:
            :param exc:
            :param tb:
            :returns:
            """

            return None

        def write(self, data: bytes) -> int:
            """
            write: Function description.
            :param data:
            :returns:
            """

            raise RuntimeError("disk full")

    monkeypatch.setattr(ingest.tempfile, "NamedTemporaryFile", lambda delete=False: _Temp())

    with pytest.raises(ingest.ArtifactDownloadError, match="Failed to stream artifact"):
        ingest._stream_to_temp_file("https://example.com/x")

    assert not (tmp_path / "temp.bin").exists()


def test_prepare_artifact_bundle_routes_by_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_prepare_artifact_bundle_routes_by_host: Function description.
    :param monkeypatch:
    :returns:
    """

    called: list[str] = []

    def fake_generic(url: str) -> ingest.ArtifactBundle:
        """
        fake_generic: Function description.
        :param url:
        :returns:
        """

        called.append(f"generic:{url}")
        return ingest.ArtifactBundle(
            kind="file",
            path=Path("dummy"),
            cleanup_root=Path("dummy"),
        )

    def fake_hf(parsed_url: Any, repo_info: Tuple[str, str]) -> ingest.ArtifactBundle:
        """
        fake_hf: Function description.
        :param parsed_url:
        :param repo_info:
        :returns:
        """

        called.append(f"hf:{repo_info[0]}:{repo_info[1]}")
        return ingest.ArtifactBundle(
            kind="file",
            path=Path("dummy"),
            cleanup_root=Path("dummy"),
        )

    def fake_gh(parsed_url: Any, repo: Tuple[str, str]) -> ingest.ArtifactBundle:
        """
        fake_gh: Function description.
        :param parsed_url:
        :param repo:
        :returns:
        """

        called.append(f"gh:{repo[0]}:{repo[1]}")
        return ingest.ArtifactBundle(
            kind="file",
            path=Path("dummy"),
            cleanup_root=Path("dummy"),
        )

    monkeypatch.setattr(ingest, "_download_generic_file", fake_generic)
    monkeypatch.setattr(ingest, "_download_huggingface_repo", fake_hf)
    monkeypatch.setattr(ingest, "_download_github_repo", fake_gh)

    ingest.prepare_artifact_bundle("https://huggingface.co/org/model")
    ingest.prepare_artifact_bundle(
        "https://huggingface.co/org/model/resolve/main/file"
    )
    ingest.prepare_artifact_bundle("https://github.com/org/repo")
    ingest.prepare_artifact_bundle("https://example.com/file.bin")

    assert called == [
        "hf:model:org/model",
        "generic:https://huggingface.co/org/model/resolve/main/file",
        "gh:org:repo",
        "generic:https://example.com/file.bin",
    ]


def test_prepare_artifact_bundle_raises_when_hf_repo_unknown() -> None:
    """
    test_prepare_artifact_bundle_raises_when_hf_repo_unknown: Function description.
    :param:
    :returns:
    """

    with pytest.raises(ingest.ArtifactDownloadError):
        ingest.prepare_artifact_bundle("https://huggingface.co/")


def test_download_huggingface_repo_creates_archive_and_readme_excerpt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_download_huggingface_repo_creates_archive_and_readme_excerpt: Function description.
    :param monkeypatch:
    :returns:
    """

    fake_files = ["README.md", "weights.bin"]

    class _FakeApi:
        """
        _FakeApi: Class description.
        """

        def list_repo_files(self, *, repo_id: str, repo_type: str) -> list[str]:
            """
            list_repo_files: Function description.
            :param repo_id:
            :param repo_type:
            :returns:
            """

            assert repo_id == "org/model"
            assert repo_type == "model"
            return list(fake_files)

    def fake_hf_hub_download(
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        cache_dir: Path,
        local_dir: Path,
    ) -> str:
        """
        fake_hf_hub_download: Function description.
        :param repo_id:
        :param filename:
        :param repo_type:
        :param cache_dir:
        :param local_dir:
        :returns:
        """

        dest = local_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = "Hello README" if filename == "README.md" else "binary"
        dest.write_text(content, encoding="utf-8")
        return str(dest)

    monkeypatch.setattr(ingest, "HfApi", _FakeApi)
    monkeypatch.setattr(ingest, "hf_hub_download", fake_hf_hub_download)

    bundle = ingest._download_huggingface_repo(
        parsed_url=None,
        repo_info=("model", "org/model"),
    )
    try:
        assert bundle.kind == "file"
        assert bundle.path.exists()
        assert bundle.readme_excerpt == "Hello README"
    finally:
        shutil.rmtree(bundle.cleanup_root, ignore_errors=True)


def test_download_huggingface_repo_empty_files_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_download_huggingface_repo_empty_files_raises: Function description.
    :param monkeypatch:
    :returns:
    """

    class _FakeApi:
        """
        _FakeApi: Class description.
        """

        def list_repo_files(self, *, repo_id: str, repo_type: str) -> list[str]:
            """
            list_repo_files: Function description.
            :param repo_id:
            :param repo_type:
            :returns:
            """

            return []

    monkeypatch.setattr(ingest, "HfApi", _FakeApi)

    with pytest.raises(ingest.ArtifactDownloadError):
        ingest._download_huggingface_repo(
            parsed_url=None,
            repo_info=("model", "x/y"),
        )


def test_download_huggingface_repo_wraps_list_repo_files_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_download_huggingface_repo_wraps_list_repo_files_errors: Function description.
    :param monkeypatch:
    :returns:
    """

    class _FakeApi:
        """
        _FakeApi: Class description.
        """

        def list_repo_files(self, *, repo_id: str, repo_type: str) -> list[str]:
            """
            list_repo_files: Function description.
            :param repo_id:
            :param repo_type:
            :returns:
            """

            raise RuntimeError("boom")

    monkeypatch.setattr(ingest, "HfApi", _FakeApi)

    with pytest.raises(ingest.ArtifactDownloadError, match="Failed to enumerate"):
        ingest._download_huggingface_repo(None, ("model", "org/model"))


def test_download_huggingface_repo_cleans_up_on_download_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_download_huggingface_repo_cleans_up_on_download_failure: Function description.
    :param monkeypatch:
    :returns:
    """

    class _FakeApi:
        """
        _FakeApi: Class description.
        """

        def list_repo_files(self, *, repo_id: str, repo_type: str) -> list[str]:
            """
            list_repo_files: Function description.
            :param repo_id:
            :param repo_type:
            :returns:
            """

            return ["README.md"]

    monkeypatch.setattr(ingest, "HfApi", _FakeApi)
    monkeypatch.setattr(ingest, "hf_hub_download", lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(ingest.ArtifactDownloadError, match="Failed to download"):
        ingest._download_huggingface_repo(None, ("model", "org/model"))


def test_readme_limit_env_invalid_is_treated_as_unlimited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_readme_limit_env_invalid_is_treated_as_unlimited: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("ARTIFACT_README_MAX_BYTES", "bogus")
    reloaded = importlib.reload(ingest)
    assert reloaded.README_CAPTURE_LIMIT is None


def test_extract_readme_from_zip_returns_none_for_bad_archive(
    tmp_path: Path,
) -> None:
    """
    test_extract_readme_from_zip_returns_none_for_bad_archive: Function description.
    :param tmp_path:
    :returns:
    """

    archive = tmp_path / "bad.zip"
    archive.write_bytes(b"not-a-zip")

    assert ingest._extract_readme_from_zip(archive) is None


def test_download_github_repo_tries_multiple_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    test_download_github_repo_tries_multiple_branches: Function description.
    :param monkeypatch:
    :param tmp_path:
    :returns:
    """

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(ingest, "_resolve_github_default_branch", lambda *_: "dev")

    calls: list[tuple[str, dict[str, str] | None]] = []

    def fake_stream(url: str, *, headers: dict[str, str] | None = None):
        """
        fake_stream: Function description.
        :param url:
        :param headers:
        :returns:
        """

        calls.append((url, headers))
        assert headers and headers.get("Authorization") == "Bearer token"
        if url.endswith("/dev"):
            raise ingest.ArtifactDownloadError("dev missing")
        path = tmp_path / "repo.zip"
        path.write_bytes(b"PK\x03\x04")
        return path, "application/zip"

    monkeypatch.setattr(ingest, "_stream_to_temp_file", fake_stream)
    monkeypatch.setattr(ingest, "_extract_readme_from_zip", lambda _: "README")

    bundle = ingest._download_github_repo(None, ("org", "repo"))

    assert bundle.kind == "file"
    assert bundle.readme_excerpt == "README"
    assert calls and calls[0][0].endswith("/dev")


def test_download_github_repo_raises_after_all_branches_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_download_github_repo_raises_after_all_branches_fail: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setattr(ingest, "_resolve_github_default_branch", lambda *_: None)

    def always_fail(*_: Any, **__: Any):
        """
        always_fail: Function description.
        :param *_:
        :param **__:
        :returns:
        """

        raise ingest.ArtifactDownloadError("nope")

    monkeypatch.setattr(ingest, "_stream_to_temp_file", always_fail)

    with pytest.raises(ingest.ArtifactDownloadError, match="Failed to download"):
        ingest._download_github_repo(None, ("org", "repo"))


def test_resolve_github_default_branch_parses_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_resolve_github_default_branch_parses_response: Function description.
    :param monkeypatch:
    :returns:
    """

    class _Response:
        """
        _Response: Class description.
        """

        status_code = 200

        def json(self) -> dict[str, str]:
            """
            json: Function description.
            :param:
            :returns:
            """

            return {"default_branch": "main"}

    monkeypatch.setattr(ingest.requests, "get", lambda *_, **__: _Response())
    assert ingest._resolve_github_default_branch("org", "repo") == "main"


def test_resolve_github_default_branch_returns_none_on_request_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_resolve_github_default_branch_returns_none_on_request_failure: Function description.
    :param monkeypatch:
    :returns:
    """

    def boom(*args: Any, **kwargs: Any) -> Any:
        """
        boom: Function description.
        :param *args:
        :param **kwargs:
        :returns:
        """

        raise RuntimeError("network")

    monkeypatch.setattr(ingest.requests, "get", boom)
    assert ingest._resolve_github_default_branch("org", "repo") is None


def test_download_generic_file_returns_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    test_download_generic_file_returns_bundle: Function description.
    :param monkeypatch:
    :param tmp_path:
    :returns:
    """

    path = tmp_path / "file.bin"
    path.write_bytes(b"x")
    monkeypatch.setattr(ingest, "_stream_to_temp_file", lambda url: (path, "application/bin"))
    bundle = ingest._download_generic_file("https://example.com/file.bin")
    assert bundle.kind == "file"
    assert bundle.path == path
    assert bundle.content_type == "application/bin"
