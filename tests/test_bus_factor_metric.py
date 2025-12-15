"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for test bus factor metric module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

import src.metrics.bus_factor as bus_factor
from src.metrics.bus_factor import BusFactorMetric


@dataclass
class _FakeHFInfo:
    """
    _FakeHFInfo: Class description.
    """

    id: str
    card_data: Dict[str, Any]
    downloads: int = 0
    likes: int = 0
    lastModified: Optional[str] = None


@dataclass
class _FakeHFClient:
    """
    _FakeHFClient: Class description.
    """

    readme: str
    model_info: Optional[_FakeHFInfo] = None

    def get_model_readme(self, repo_id: str) -> str:
        """
        get_model_readme: Function description.
        :param repo_id:
        :returns:
        """

        return self.readme

    def get_model_info(self, repo_id: str) -> Any:
        """
        get_model_info: Function description.
        :param repo_id:
        :returns:
        """

        if self.model_info is None:
            raise RuntimeError("model info missing")
        return self.model_info


class _FakeGitClient:
    """
    _FakeGitClient: Class description.
    """

    def __init__(
        self,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        contributors: Optional[List[Dict[str, Any]]] = None,
        fail_metadata: bool = False,
        fail_contributors: bool = False,
    ) -> None:
        """
        __init__: Function description.
        :param metadata:
        :param contributors:
        :param fail_metadata:
        :param fail_contributors:
        :returns:
        """

        self._metadata = metadata
        self._contributors = contributors or []
        self._fail_metadata = fail_metadata
        self._fail_contributors = fail_contributors
        self.metadata_calls: List[str] = []
        self.contributor_calls: List[str] = []

    def get_repo_metadata(self, repo_url: str) -> Dict[str, Any]:
        """
        get_repo_metadata: Function description.
        :param repo_url:
        :returns:
        """

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
        """
        list_repo_files: Function description.
        :param repo_url:
        :param branch:
        :returns:
        """

        return []

    def list_repo_contributors(
        self,
        repo_url: str,
        *,
        per_page: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        list_repo_contributors: Function description.
        :param repo_url:
        :param per_page:
        :returns:
        """

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
    """
    _metric: Function description.
    :param readme:
    :param git_client:
    :param hf_info:
    :returns:
    """

    return BusFactorMetric(
        hf_client=_FakeHFClient(readme=readme, model_info=hf_info),
        git_client=git_client,
    )


def test_bus_factor_high_with_many_contributors() -> None:
    """
    test_bus_factor_high_with_many_contributors: Function description.
    :param:
    :returns:
    """

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
    """
    test_bus_factor_low_for_single_contributor: Function description.
    :param:
    :returns:
    """

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
    """
    test_bus_factor_readme_fallback: Function description.
    :param:
    :returns:
    """

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
    """
    test_bus_factor_handles_git_failures: Function description.
    :param:
    :returns:
    """

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
    """
    test_bus_factor_hf_metadata_fallback: Function description.
    :param:
    :returns:
    """

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


def test_bus_factor_select_repo_url_respects_readme_fallback_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_bus_factor_select_repo_url_respects_readme_fallback_toggle: Function description.
    :param monkeypatch:
    :returns:
    """

    metric = _metric(readme="", git_client=_FakeGitClient(metadata=None))
    monkeypatch.setattr(bus_factor, "enable_readme_fallback", lambda: False)

    assert (
        metric._select_repo_url(
            {"hf_url": "https://huggingface.co/org/model"},
            "See https://github.com/org/repo for code",
        )
        is None
    )


def test_bus_factor_safe_helpers_swallow_exceptions() -> None:
    """
    test_bus_factor_safe_helpers_swallow_exceptions: Function description.
    :param:
    :returns:
    """

    metric = _metric(
        readme="",
        git_client=_FakeGitClient(metadata=None, fail_metadata=True, fail_contributors=True),
    )

    assert metric._safe_repo_metadata("https://github.com/org/repo") is None
    assert metric._safe_contributors("https://github.com/org/repo") == []
    assert metric._safe_model_info("https://huggingface.co/org/model") is None

    class ExplodingHFClient(_FakeHFClient):
        """
        ExplodingHFClient: Class description.
        """

        def get_model_readme(self, repo_id: str) -> str:
            """
            get_model_readme: Function description.
            :param repo_id:
            :returns:
            """

            raise RuntimeError("boom")

    metric._hf = ExplodingHFClient(readme="", model_info=_FakeHFInfo(id="x", card_data={}))
    assert metric._safe_readme("https://huggingface.co/org/model") == ""


def test_contributor_diversity_branches() -> None:
    """
    test_contributor_diversity_branches: Function description.
    :param:
    :returns:
    """

    assert bus_factor._contributor_diversity([]) == 0.0
    assert bus_factor._contributor_diversity([{"contributions": 0}]) == 0.0
    assert bus_factor._contributor_diversity(
        [{"contributions": 10}, {"contributions": 5}]
    ) > 0.0
    assert bus_factor._contributor_diversity(
        [{"contributions": 10}, {"contributions": 5}, {"contributions": 1}]
    ) > 0.0


def test_ownership_resilience_branches() -> None:
    """
    test_ownership_resilience_branches: Function description.
    :param:
    :returns:
    """

    assert bus_factor._ownership_resilience(None, [{"x": 1}]) == 0.1
    assert bus_factor._ownership_resilience(None, [{"x": 1}, {"x": 2}]) == 0.2
    assert (
        bus_factor._ownership_resilience(
            {"archived": True, "owner": {"type": "Organization"}},
            [{"x": 1}, {"x": 2}, {"x": 3}],
        )
        == 0.0
    )
    assert (
        bus_factor._ownership_resilience(
            {"archived": False, "owner": {"type": "Organization"}},
            [],
        )
        == 1.0
    )
    assert (
        bus_factor._ownership_resilience(
            {"archived": False, "owner": {"type": "User"}, "fork": True},
            [{"x": 1}],
        )
        == 0.2
    )


def test_community_support_branches() -> None:
    """
    test_community_support_branches: Function description.
    :param:
    :returns:
    """

    now = "2025-01-01T00:00:00+00:00"
    assert (
        bus_factor._community_support(
            {"stargazers_count": "bad", "watchers_count": 5000, "pushed_at": now},
            None,
        )
        > 0.0
    )

    @dataclass
    class HFInfo:
        """
        HFInfo: Class description.
        """

        card_data: Dict[str, Any]
        downloads: int = 10
        likes: int = 0
        lastModified: str = now

    assert bus_factor._community_support(None, HFInfo(card_data={"deprecated": True})) == 0.0


def test_hf_metadata_fallback_ownership_score_branches() -> None:
    @dataclass
    class HFInfo:
        """
        HFInfo: Class description.
        """

        id: str
        card_data: Dict[str, Any]
        downloads: int = 0
        likes: int = 0
        lastModified: str = "2025-01-01T00:00:00+00:00"

    assert bus_factor._hf_metadata_fallback(HFInfo(id="google/model", card_data={})) is not None
    assert (
        bus_factor._hf_metadata_fallback(
            HFInfo(id="user/model", card_data={"maintainers": ["a", "b", "c"]})
        )
        is not None
    )
    assert (
        bus_factor._hf_metadata_fallback(
            HFInfo(id="user/model", card_data={"maintainers": ["a", "b"]})
        )
        is not None
    )


def test_extract_readme_maintainers_toggle_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_extract_readme_maintainers_toggle_and_fallback: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setattr(bus_factor, "enable_readme_fallback", lambda: False)
    assert bus_factor._extract_readme_maintainers("## Maintainers\n- A") == []

    monkeypatch.setattr(bus_factor, "enable_readme_fallback", lambda: True)
    readme = "Text\n- Alice (maintainer)\n- Bob (author)\n"
    assert bus_factor._extract_readme_maintainers(readme) == [
        "Alice (maintainer)",
        "Bob (author)",
    ]


def test_readme_contributor_score_branches() -> None:
    """
    test_readme_contributor_score_branches: Function description.
    :param:
    :returns:
    """

    assert bus_factor._readme_contributor_score([]) == 0.0
    assert bus_factor._readme_contributor_score(["a"]) == 0.2
    assert bus_factor._readme_contributor_score(["a", "b"]) == 0.4
    assert bus_factor._readme_contributor_score(["a", "b", "c"]) == 0.6
    assert bus_factor._readme_contributor_score(["a", "b", "c", "d", "e"]) == 0.8


def test_bus_factor_fail_stub_returns_expected_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_bus_factor_fail_stub_returns_expected_value: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("ACME_IGNORE_FAIL", "0")
    monkeypatch.setattr(bus_factor, "FAIL", True)
    metric = BusFactorMetric(
        hf_client=_FakeHFClient(readme=""),
        git_client=_FakeGitClient(),
    )

    assert metric.compute(
        {"hf_url": "https://huggingface.co/parvk11/audience_classifier_model"}
    ) == pytest.approx(0.7)


def test_bus_factor_returns_zero_without_hf_url() -> None:
    """
    test_bus_factor_returns_zero_without_hf_url: Function description.
    :param:
    :returns:
    """

    metric = BusFactorMetric(hf_client=_FakeHFClient(readme=""), git_client=_FakeGitClient())
    assert metric.compute({"hf_url": ""}) == pytest.approx(0.0)


def test_bus_factor_helper_functions_cover_edge_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_bus_factor_helper_functions_cover_edge_cases: Function description.
    :param monkeypatch:
    :returns:
    """

    assert bus_factor._normalize_github_url("https://example.com/x") is None
    assert bus_factor._normalize_github_url("https://github.com/org/repo/issues") == (
        "https://github.com/org/repo"
    )

    assert bus_factor._contributor_diversity([]) == 0.0
    assert bus_factor._contributor_diversity([{"contributions": 0}]) == 0.0

    assert bus_factor._ownership_resilience(None, [{"c": 1}]) == 0.1
    assert bus_factor._ownership_resilience(None, [{"c": 1}, {"c": 2}]) == 0.2
    assert bus_factor._ownership_resilience({"archived": True}, []) == 0.0
    assert (
        bus_factor._ownership_resilience(
            {"owner": {"type": "Organization"}, "archived": False},
            [{"c": 1}],
        )
        == 1.0
    )
    assert (
        bus_factor._ownership_resilience(
            {"owner": {"type": "User"}, "archived": False},
            [{"c": 1}, {"c": 2}, {"c": 3}],
        )
        == 0.6
    )
    assert (
        bus_factor._ownership_resilience(
            {"owner": {"type": "User"}, "archived": False, "fork": True},
            [{"c": 1}],
        )
        == 0.2
    )

    assert bus_factor._community_support({"archived": True}, None) == 0.0
    assert (
        bus_factor._community_support(
            {
                "archived": False,
                "stargazers_count": 1000000,
                "pushed_at": "2024-01-01T00:00:00Z",
            },
            None,
        )
        >= 0.5
    )

    class HFInfo:
        """
        HFInfo: Class description.
        """

        downloads = 12345
        likes = 0
        lastModified = "2024-01-01T00:00:00Z"  # noqa: N815 - upstream schema
        card_data = {"deprecated": False}

    assert bus_factor._community_support(None, HFInfo()) > 0.0
    assert bus_factor._hf_metadata_fallback(None) is None

    monkeypatch.setenv("ACME_ENABLE_README_FALLBACK", "0")
    assert bus_factor._extract_readme_maintainers("## Maintainers\n- Alice") == []

    monkeypatch.setenv("ACME_ENABLE_README_FALLBACK", "1")
    text = "- Alice (maintainer)\n- Bob\n"
    assert bus_factor._extract_readme_maintainers(text) == ["Alice (maintainer)"]

    assert bus_factor._days_since("not-a-date") is None
