"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for tree score metric module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, cast

import pytest

from src.clients.hf_client import HFClient
from src.metrics import tree_score


@dataclass
class _FakeModelInfo:
    """
    _FakeModelInfo: Class description.
    """

    card_data: Optional[Mapping[str, Any]] = None
    config: Optional[Mapping[str, Any]] = None
    base_model: Optional[str] = None
    base_model_id: Optional[str] = None
    parent_model: Optional[str] = None
    parent_models: Iterable[str] = field(default_factory=list)
    parents: Iterable[str] = field(default_factory=list)


class _FakeHFClient:
    """
    _FakeHFClient: Class description.
    """

    def __init__(
        self,
        *,
        info: Any = None,
        readme: str = "",
        info_error: Optional[Exception] = None,
        readme_error: Optional[Exception] = None,
    ) -> None:
        """
        __init__: Function description.
        :param info:
        :param readme:
        :param info_error:
        :param readme_error:
        :returns:
        """

        self._info = info
        self._readme = readme
        self._info_error = info_error
        self._readme_error = readme_error
        self.info_calls: List[str] = []
        self.readme_calls: List[str] = []

    def get_model_info(self, repo_id: str) -> Any:
        """
        get_model_info: Function description.
        :param repo_id:
        :returns:
        """

        self.info_calls.append(repo_id)
        if self._info_error is not None:
            raise self._info_error
        return self._info

    def get_model_readme(self, repo_id: str) -> str:
        """
        get_model_readme: Function description.
        :param repo_id:
        :returns:
        """

        self.readme_calls.append(repo_id)
        if self._readme_error is not None:
            raise self._readme_error
        return self._readme


def _metric(hf: Optional[_FakeHFClient] = None) -> tree_score.TreeScoreMetric:
    """
    _metric: Function description.
    :param hf:
    :returns:
    """

    client = cast(HFClient, hf or _FakeHFClient())
    return tree_score.TreeScoreMetric(hf_client=client)


def test_tree_score_fail_stub_returns_known_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_tree_score_fail_stub_returns_known_value: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.delenv("ACME_IGNORE_FAIL", raising=False)
    metric = _metric()

    monkeypatch.setattr(tree_score, "FAIL", True, raising=False)

    value = metric.compute(
        {"hf_url": "https://huggingface.co/google-bert/bert-base-uncased"}
    )

    assert value == pytest.approx(0.6)
    assert metric._hf.info_calls == []  # type: ignore[attr-defined]
    assert metric._hf.readme_calls == []  # type: ignore[attr-defined]


def test_tree_score_missing_url_defaults_to_half() -> None:
    """
    test_tree_score_missing_url_defaults_to_half: Function description.
    :param:
    :returns:
    """

    metric = _metric()

    value = metric.compute({})

    assert value == pytest.approx(0.5)


def test_tree_score_no_parents_falls_back_to_target_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_tree_score_no_parents_falls_back_to_target_score: Function description.
    :param monkeypatch:
    :returns:
    """

    calls: List[str] = []

    def _fake_parent_score(slug: str) -> Optional[float]:
        """
        _fake_parent_score: Function description.
        :param slug:
        :returns:
        """

        calls.append(slug)
        return 0.42

    monkeypatch.setattr(tree_score, "_discover_parents", lambda *_, **__: [])
    monkeypatch.setattr(
        tree_score,
        "_compute_parent_net_score",
        _fake_parent_score,
    )
    metric = _metric()

    value = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert value == pytest.approx(0.42)
    assert calls == ["org/model"]


def test_tree_score_no_ancestor_scores_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_tree_score_no_ancestor_scores_falls_back_to_default: Function description.
    :param monkeypatch:
    :returns:
    """

    chains = {"parent/model": {"parent/model": 0, "upstream/root": 1}}
    compute_calls: List[str] = []

    def _fake_collect(
        _hf: Any, slug: str, _depth: int, visited: Optional[set[str]] = None
    ) -> Dict[str, int]:
        """
        _fake_collect: Function description.
        :param _hf:
        :param slug:
        :param _depth:
        :param visited:
        :returns:
        """

        if visited is not None:
            visited.update(chains[slug].keys())
        return dict(chains[slug])

    def _fake_score(slug: str) -> Optional[float]:
        """
        _fake_score: Function description.
        :param slug:
        :returns:
        """

        compute_calls.append(slug)
        return None

    monkeypatch.setattr(
        tree_score, "_discover_parents", lambda *_: ["parent/model"]
    )
    monkeypatch.setattr(
        tree_score,
        "_collect_ancestors_with_depth",
        _fake_collect,
    )
    monkeypatch.setattr(tree_score, "_compute_parent_net_score", _fake_score)
    metric = _metric()

    value = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert value == pytest.approx(0.5)
    assert compute_calls == [
        "parent/model",
        "upstream/root",
        "org/model",
    ]


def test_tree_score_success_weights_ancestors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_tree_score_success_weights_ancestors: Function description.
    :param monkeypatch:
    :returns:
    """

    chains = {
        "owner/base": {"owner/base": 0, "shared/root": 1},
        "owner/alt": {
            "owner/alt": 0,
            "shared/root": 2,
            "owner/grand": 2,
        },
    }
    scores = {
        "owner/base": 0.8,
        "shared/root": 0.6,
        "owner/alt": 0.4,
        "owner/grand": 0.9,
    }

    def _fake_collect(
        _hf: Any, slug: str, _depth: int, visited: Optional[set[str]] = None
    ) -> Dict[str, int]:
        """
        _fake_collect: Function description.
        :param _hf:
        :param slug:
        :param _depth:
        :param visited:
        :returns:
        """

        if visited is not None:
            visited.update(chains[slug].keys())
        return dict(chains[slug])

    def _fake_score(slug: str) -> Optional[float]:
        """
        _fake_score: Function description.
        :param slug:
        :returns:
        """

        return scores.get(slug)

    monkeypatch.setattr(
        tree_score,
        "_discover_parents",
        lambda *_: ["owner/base", "owner/alt"],
    )
    monkeypatch.setattr(
        tree_score,
        "_collect_ancestors_with_depth",
        _fake_collect,
    )
    monkeypatch.setattr(tree_score, "_compute_parent_net_score", _fake_score)
    metric = _metric()

    value = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    # Weighted average: base(1.0), alt(1.0), shared(0.7), grand(0.5)
    expected = (0.8 + 0.4 + 0.6 * 0.7 + 0.9 * 0.5) / (1.0 + 1.0 + 0.7 + 0.5)
    assert value == pytest.approx(expected)


def test_collect_ancestors_with_depth_handles_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_collect_ancestors_with_depth_handles_cycles: Function description.
    :param monkeypatch:
    :returns:
    """

    graph = {
        "root/base": ["loop/parent"],
        "loop/parent": ["root/base", "leaf/model"],
        "leaf/model": ["deep/model"],
        "deep/model": ["ignored/node"],
        "ignored/node": [],
    }

    def _fake_discover(_hf: Any, url: str) -> List[str]:
        """
        _fake_discover: Function description.
        :param _hf:
        :param url:
        :returns:
        """

        slug = url.removeprefix("https://huggingface.co/")
        return graph.get(slug, [])

    monkeypatch.setattr(tree_score, "_discover_parents", _fake_discover)
    visited = {"seed/model"}

    result = tree_score._collect_ancestors_with_depth(
        cast(HFClient, _FakeHFClient()),
        "root/base",
        max_depth=2,
        visited=visited,
    )

    assert result == {
        "root/base": 0,
        "loop/parent": 1,
        "leaf/model": 2,
    }
    assert "seed/model" in visited


def test_discover_parents_aggregates_metadata_and_readme() -> None:
    """
    test_discover_parents_aggregates_metadata_and_readme: Function description.
    :param:
    :returns:
    """

    info = _FakeModelInfo(
        card_data={
            "base_model": "owner/base",
            "parent_models": ["owner/alt", "owner/base"],
        },
        config={"parent_model": "owner/third"},
        parent_models=["owner/fourth"],
    )
    readme = """
    ## Model Card
    Base model: owner/fifth
    Fine-tuned from owner/fifth on new data.
    Additional info at https://huggingface.co/owner/sixth-model
    """
    hf = _FakeHFClient(info=info, readme=readme)

    parents = tree_score._discover_parents(
        cast(HFClient, hf),
        "https://huggingface.co/org/model",
    )

    assert parents == [
        "owner/base",
        "owner/alt",
        "owner/third",
        "owner/fourth",
        "owner/fifth",
    ]
    assert hf.info_calls == ["https://huggingface.co/org/model"]
    assert hf.readme_calls == ["https://huggingface.co/org/model"]


def test_discover_parents_handles_failures() -> None:
    """
    test_discover_parents_handles_failures: Function description.
    :param:
    :returns:
    """

    hf = _FakeHFClient(
        info_error=RuntimeError("fail info"),
        readme_error=RuntimeError("fail readme"),
    )

    parents = tree_score._discover_parents(
        cast(HFClient, hf),
        "https://huggingface.co/org/model",
    )

    assert parents == []


def test_extract_helpers_cover_edge_cases() -> None:
    """
    test_extract_helpers_cover_edge_cases: Function description.
    :param:
    :returns:
    """

    mapping = {
        "base_model": "owner/base",
        "parent_models": ["owner/alt", b"owner/base"],
        "parents": ("owner/third", 42),
    }

    class _Obj:
        """
        _Obj: Class description.
        """

        base_model = "owner/base"
        parent_model = ["owner/fourth", b"owner/fifth"]

    readme = """
    parent model: owner/sixth
    fine-tuned from owner/seventh
    Check https://huggingface.co/owner/eighth
    """

    assert tree_score._extract_parents_from_mapping(mapping) == [
        "owner/base",
        "owner/third",
        "owner/alt",
        "b'owner/base'",
    ]
    assert tree_score._extract_parents_from_object(_Obj()) == [
        "owner/base",
        "owner/fourth",
        "b'owner/fifth'",
    ]
    assert tree_score._extract_parents_from_readme(readme) == [
        "owner/sixth",
        "owner/seventh",
        "owner/eighth",
    ]


def test_extract_hf_url_and_slug_helpers() -> None:
    """
    test_extract_hf_url_and_slug_helpers: Function description.
    :param:
    :returns:
    """

    record = {"hf_url": "https://huggingface.co/owner/model"}
    assert tree_score._extract_hf_url(record) == record["hf_url"]
    assert tree_score._extract_hf_url({"hf_url": 123}) is None

    assert tree_score._to_repo_slug("owner/model/extra") == "owner/model"
    assert tree_score._to_repo_slug(
        "https://huggingface.co/models/owner/model/tree/main"
    ) == "owner/model"
    assert tree_score._to_repo_slug("invalid") is None


def test_to_numeric_metric_variants() -> None:
    """
    test_to_numeric_metric_variants: Function description.
    :param:
    :returns:
    """

    assert tree_score._to_numeric_metric(1) == 1.0
    assert tree_score._to_numeric_metric(
        {"desktop_pc": 0.9, "edge": 0.4}
    ) == 0.9
    assert tree_score._to_numeric_metric({"mobile": 0.3, "tablet": 0.7}) == 0.5
    assert tree_score._to_numeric_metric(
        cast(Any, {"mobile": "n/a"})
    ) is None
    assert tree_score._to_numeric_metric(cast(Any, "not numeric")) is None


def test_compute_parent_net_score_averages_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_compute_parent_net_score_averages_metrics: Function description.
    :param monkeypatch:
    :returns:
    """

    class _StubMetric:
        """
        _StubMetric: Class description.
        """

        def __init__(self, name: str, output: Any) -> None:
            """
            __init__: Function description.
            :param name:
            :param output:
            :returns:
            """

            self.name = name
            self.key = name
            self._output = output

        def compute(self, _record: Mapping[str, str]) -> Any:
            """
            compute: Function description.
            :param _record:
            :returns:
            """

            if isinstance(self._output, Exception):
                raise self._output
            return self._output

    def _factory(name: str, output: Any) -> Any:
        """
        _factory: Function description.
        :param name:
        :param output:
        :returns:
        """

        return _StubMetric(name, output)

    monkeypatch.setattr(
        tree_score, "RampUpMetric", lambda: _factory("Ramp Up", 0.2)
    )
    monkeypatch.setattr(
        tree_score,
        "BusFactorMetric",
        lambda: _factory("Bus Factor", {"desktop_pc": 0.4}),
    )
    monkeypatch.setattr(
        tree_score,
        "LicenseMetric",
        lambda: _factory("License", {"a": 0.6, "b": 0.8}),
    )
    monkeypatch.setattr(
        tree_score,
        "SizeMetric",
        lambda: _factory("Size", {"small": 1.0, "large": 2.0}),
    )
    monkeypatch.setattr(
        tree_score,
        "DatasetAndCodeMetric",
        lambda: _factory("Dataset & Code", RuntimeError("fail")),
    )
    monkeypatch.setattr(
        tree_score,
        "DatasetQualityMetric",
        lambda: _factory("Dataset Quality", 0.9),
    )
    monkeypatch.setattr(
        tree_score,
        "CodeQualityMetric",
        lambda: _factory("Code Quality", {"desktop_pc": 0.5, "mobile": 0.2}),
    )
    monkeypatch.setattr(
        tree_score,
        "PerformanceMetric",
        lambda: _factory("Performance", 0.3),
    )

    score = tree_score._compute_parent_net_score("owner/model")

    assert score == pytest.approx(0.642857, rel=1e-6)


def test_compute_parent_net_score_returns_none_when_no_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_compute_parent_net_score_returns_none_when_no_values: Function description.
    :param monkeypatch:
    :returns:
    """

    class _StubMetric:
        """
        _StubMetric: Class description.
        """

        def __init__(self, name: str, output: Any) -> None:
            """
            __init__: Function description.
            :param name:
            :param output:
            :returns:
            """

            self.name = name
            self.key = name
            self._output = output

        def compute(self, _record: Mapping[str, str]) -> Any:
            """
            compute: Function description.
            :param _record:
            :returns:
            """

            if isinstance(self._output, Exception):
                raise self._output
            return self._output

    def _factory(name: str, output: Any) -> Any:
        """
        _factory: Function description.
        :param name:
        :param output:
        :returns:
        """

        return _StubMetric(name, output)

    monkeypatch.setattr(
        tree_score, "RampUpMetric", lambda: _factory("Ramp Up", "n/a")
    )
    monkeypatch.setattr(
        tree_score,
        "BusFactorMetric",
        lambda: _factory("Bus Factor", RuntimeError("fail")),
    )
    monkeypatch.setattr(
        tree_score,
        "LicenseMetric",
        lambda: _factory("License", {"a": "x"}),
    )
    monkeypatch.setattr(
        tree_score, "SizeMetric", lambda: _factory("Size", {})
    )
    monkeypatch.setattr(
        tree_score,
        "DatasetAndCodeMetric",
        lambda: _factory("Dataset & Code", RuntimeError("fail")),
    )
    monkeypatch.setattr(
        tree_score,
        "DatasetQualityMetric",
        lambda: _factory("Dataset Quality", RuntimeError("fail")),
    )
    monkeypatch.setattr(
        tree_score,
        "CodeQualityMetric",
        lambda: _factory("Code Quality", {"desktop_pc": "y"}),
    )
    monkeypatch.setattr(
        tree_score,
        "PerformanceMetric",
        lambda: _factory("Performance", RuntimeError("fail")),
    )

    score = tree_score._compute_parent_net_score("owner/model")

    assert score is None
