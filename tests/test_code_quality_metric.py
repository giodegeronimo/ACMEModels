"""Tests for test code quality metric module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from src.metrics.code_quality import CodeQualityMetric


@dataclass
class _FakeHFClient:
    readme: str

    def get_model_readme(self, repo_id: str) -> str:
        return self.readme


class _FakeGitClient:
    def __init__(
        self,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        files: Optional[List[str]] = None,
        fail_tree: bool = False,
    ) -> None:
        self._metadata = metadata
        self._files = files or []
        self._fail_tree = fail_tree
        self.metadata_calls: List[str] = []
        self.tree_calls: List[str] = []

    def get_repo_metadata(self, repo_url: str) -> Dict[str, Any]:
        self.metadata_calls.append(repo_url)
        if self._metadata is None:
            raise RuntimeError("metadata missing")
        return dict(self._metadata)

    def list_repo_files(
        self,
        repo_url: str,
        *,
        branch: Optional[str] = None,
    ) -> List[str]:
        self.tree_calls.append(f"{repo_url}@{branch}")
        if self._fail_tree:
            raise RuntimeError("tree missing")
        return list(self._files)


def _metric(readme: str, git_client: _FakeGitClient) -> CodeQualityMetric:
    return CodeQualityMetric(
        hf_client=_FakeHFClient(readme=readme),
        git_client=git_client,
    )


def test_code_quality_scores_high_with_repo_signals() -> None:
    readme = (
        "# Project\n"
        "## Installation\nRun the installer.\n"
        "## Usage\nExample code blocks.\n"
        "## Testing\npytest -q\n"
        "## Contributing\nSubmit PRs.\n"
        + "More documentation\n" * 100
    )
    metadata = {
        "default_branch": "main",
        "stargazers_count": 420,
        "forks_count": 50,
        "pushed_at": "2024-08-01T10:00:00Z",
    }
    files = [
        "tests/test_app.py",
        ".github/workflows/ci.yml",
        ".pre-commit-config.yaml",
        "mypy.ini",
        "docs/index.md",
        "CONTRIBUTING.md",
        "src/app.py",
    ]

    metric = _metric(readme, _FakeGitClient(metadata=metadata, files=files))

    score = metric.compute(
        {
            "hf_url": "https://huggingface.co/org/model",
            "git_url": "https://github.com/org/project",
        }
    )

    assert isinstance(score, float)
    assert score == pytest.approx(0.9, rel=0.2)


def test_code_quality_uses_readme_repo_link() -> None:
    readme = (
        "Code at https://github.com/example/repo\n"
        "## Usage\nDetails\n"
    )
    metadata = {"default_branch": "main"}
    files = ["src/main.py", "tests/test_core.py"]
    metric = _metric(readme, _FakeGitClient(metadata=metadata, files=files))

    score = metric.compute(
        {"hf_url": "https://huggingface.co/example/model"}
    )

    assert isinstance(score, float)
    assert score > 0.3


def test_code_quality_handles_missing_repo() -> None:
    readme = "Short README"
    metric = _metric(readme, _FakeGitClient(metadata=None, files=[]))

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score < 0.05


def test_code_quality_handles_git_failures() -> None:
    readme = (
        "## Testing\nDocumented\n"
        "## Contributing\nGuidelines\n"
        + "More text\n" * 50
    )
    git_client = _FakeGitClient(metadata=None, files=[], fail_tree=True)
    metric = _metric(readme, git_client)

    score = metric.compute(
        {
            "hf_url": "https://huggingface.co/org/model",
            "git_url": "https://github.com/org/repo",
        }
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 0.6
