"""Unit tests for artifact ingest helpers."""

from __future__ import annotations

import zipfile
from pathlib import Path

from src.storage import artifact_ingest as ingest


def test_extract_readme_from_directory(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("Hello directory README", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "other.txt").write_text("ignored", encoding="utf-8")

    excerpt = ingest._extract_readme_from_directory(tmp_path)

    assert excerpt == "Hello directory README"


def test_extract_readme_from_zip(tmp_path: Path) -> None:
    archive = tmp_path / "repo.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("repo/README.txt", "Zip README content")
        zf.writestr("repo/other.md", "ignored")

    excerpt = ingest._extract_readme_from_zip(archive)

    assert excerpt == "Zip README content"


def test_extract_readme_directory_preserves_full_content(
    tmp_path: Path,
) -> None:
    readme = tmp_path / "README"
    readme.write_text("A" * 5000 + "tail", encoding="utf-8")

    excerpt = ingest._extract_readme_from_directory(tmp_path)

    assert excerpt is not None
    assert excerpt.endswith("tail")


def test_extract_readme_zip_preserves_full_content(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "repo.zip"
    content = "B" * 7000 + "end"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("repo/README.md", content)

    excerpt = ingest._extract_readme_from_zip(archive)

    assert excerpt is not None
    assert excerpt.endswith("end")
