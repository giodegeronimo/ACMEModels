from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from src.metrics.bus_factor import BusFactorMetric


@dataclass
class _FakeHFInfo:
    id: str
    card_data: Dict[str, Any]
    downloads: int = 0
    likes: int = 0
    lastModified: Optional[str] = None


@dataclass
class _FakeHFClient:
    readme: str
    model_info: Optional[_FakeHFInfo] = None

    def get_model_readme(self, repo_id: str) -> str:
        return self.readme

    def get_model_info(self, repo_id: str) -> Any:
        if self.model_info is None:
            raise RuntimeError("model info missing")
        return self.model_info


class _FakeGitClient:
    def __init__(
        self,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        contributors: Optional[List[Dict[str, Any]]] = None,
        fail_metadata: bool = False,
        fail_contributors: bool = False,
    ) -> None:
        self._metadata = metadata
        self._contributors = contributors or []
        self._fail_metadata = fail_metadata
        self._fail_contributors = fail_contributors
        self.metadata_calls: List[str] = []
        self.contributor_calls: List[str] = []

    def get_repo_metadata(self, repo_url: str) -> Dict[str, Any]:
        self.metadata_calls.append(repo_url)
        if self._fail_metadata or self._metadata is None:
            raise RuntimeError("metadata unavailable")
        return dict(self._metadata)

    def list_repo_files(
        self,
        repo_url: str,
        *,
        branch: Optional[str] = None,
    ) -> List[str]:
        return []

    def list_repo_contributors(
        self,
        repo_url: str,
        *,
        per_page: int = 100,
    ) -> List[Dict[str, Any]]:
        self.contributor_calls.append(repo_url)
        if self._fail_contributors:
            raise RuntimeError("contributors unavailable")
        return list(self._contributors)


def _metric(
    readme: str,
    git_client: _FakeGitClient,
    *,
    hf_info: Optional[_FakeHFInfo] = None,
) -> BusFactorMetric:
    return BusFactorMetric(
        hf_client=_FakeHFClient(readme=readme, model_info=hf_info),
        git_client=git_client,
    )


def test_bus_factor_high_with_many_contributors() -> None:
    metadata = {
        "owner": {"type": "Organization"},
        "stargazers_count": 600,
        "pushed_at": "2024-09-01T12:00:00Z",
        "archived": False,
    }
    contributors = [
        {"login": "alice", "contributions": 120},
        {"login": "bob", "contributions": 80},
        {"login": "carol", "contributions": 60},
        {"login": "dave", "contributions": 40},
        {"login": "erin", "contributions": 25},
    ]
    metric = _metric(
        readme="",
        git_client=_FakeGitClient(
            metadata=metadata,
            contributors=contributors,
        ),
    )

    score = metric.compute(
        {
            "hf_url": "https://huggingface.co/org/model",
            "git_url": "https://github.com/org/repo",
        }
    )

    assert isinstance(score, float)
    assert score == pytest.approx(0.85, rel=0.25)


def test_bus_factor_low_for_single_contributor() -> None:
    metadata = {
        "owner": {"type": "User"},
        "stargazers_count": 5,
        "pushed_at": "2023-01-01T00:00:00Z",
        "archived": False,
    }
    contributors = [{"login": "solo", "contributions": 200}]
    metric = _metric(
        readme="",
        git_client=_FakeGitClient(
            metadata=metadata,
            contributors=contributors,
        ),
    )

    score = metric.compute(
        {
            "hf_url": "https://huggingface.co/org/model",
            "git_url": "https://github.com/org/repo",
        }
    )

    assert isinstance(score, float)
    assert score < 0.3


def test_bus_factor_readme_fallback() -> None:
    readme = (
        "# Model\n"
        "## Maintainers\n"
        "- Alice\n"
        "- Bob\n"
        "- Carol\n"
    )
    metric = _metric(
        readme,
        _FakeGitClient(metadata=None, contributors=[]),
    )

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert 0.3 <= score <= 0.6


def test_bus_factor_handles_git_failures() -> None:
    metadata = {
        "owner": {"type": "User"},
        "stargazers_count": 50,
        "pushed_at": "2022-10-10T00:00:00Z",
    }
    metric = _metric(
        readme="",
        git_client=_FakeGitClient(
            metadata=metadata,
            contributors=[],
            fail_contributors=True,
        ),
    )

    score = metric.compute(
        {
            "hf_url": "https://huggingface.co/org/model",
            "git_url": "https://github.com/org/repo",
        }
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 0.5


def test_bus_factor_hf_metadata_fallback() -> None:
    hf_info = _FakeHFInfo(
        id="org/model",
        card_data={
            "maintainers": [
                {"name": "Alice"},
                {"name": "Bob"},
                {"name": "Carol"},
            ]
        },
        downloads=150000,
        likes=800,
        lastModified="2024-09-15T12:00:00Z",
    )
    metric = _metric(
        readme="",
        git_client=_FakeGitClient(metadata=None, contributors=[]),
        hf_info=hf_info,
    )

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score > 0.5
