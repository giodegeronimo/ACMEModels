"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Unit tests for lineage extraction helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

from src.storage import lineage_extractor


def test_extract_parents_from_mapping() -> None:
    """
    test_extract_parents_from_mapping: Function description.
    :param:
    :returns:
    """

    mapping = {
        "base_model": "org/base",
        "parent_models": ["org/parent1", "org/parent2"],
        "ignored": "value",
    }

    parents = lineage_extractor._extract_parents_from_mapping(mapping)

    assert "org/base" in parents
    assert "org/parent1" in parents
    assert "org/parent2" in parents


def test_extract_parents_from_object() -> None:
    @dataclass
    class _Obj:
        """
        _Obj: Class description.
        """

        base_model: str
        parents: list[str]

    parents = lineage_extractor._extract_parents_from_object(
        _Obj(base_model="org/base", parents=["org/a", "org/b"])
    )

    assert parents == ["org/base", "org/a", "org/b"]


def test_extract_parents_from_readme_patterns() -> None:
    """
    test_extract_parents_from_readme_patterns: Function description.
    :param:
    :returns:
    """

    text = """
    # Card
    Base model: org/base
    Fine-tuned from org/parent
    """
    parents = lineage_extractor._extract_parents_from_readme(text)
    assert parents == ["org/base", "org/parent"]


def test_to_repo_slug_normalizes_urls() -> None:
    """
    test_to_repo_slug_normalizes_urls: Function description.
    :param:
    :returns:
    """

    assert lineage_extractor._to_repo_slug("org/model") == "org/model"
    assert (
        lineage_extractor._to_repo_slug("https://huggingface.co/org/model")
        == "org/model"
    )
    assert lineage_extractor._to_repo_slug("") is None


def test_extract_name_from_url_falls_back_for_invalid_urls() -> None:
    """
    test_extract_name_from_url_falls_back_for_invalid_urls: Function description.
    :param:
    :returns:
    """

    assert lineage_extractor._extract_name_from_url(
        "https://huggingface.co/org/model"
    ) == "model"
    assert lineage_extractor._extract_name_from_url("not a url") == "not a url"


def test_generate_parent_id_is_deterministic() -> None:
    """
    test_generate_parent_id_is_deterministic: Function description.
    :param:
    :returns:
    """

    assert lineage_extractor._generate_parent_id("org/model") == lineage_extractor._generate_parent_id(
        "org/model"
    )
    assert lineage_extractor._generate_parent_id("org/model").startswith("parent-")


def test_determine_relationship() -> None:
    """
    test_determine_relationship: Function description.
    :param:
    :returns:
    """

    assert lineage_extractor._determine_relationship("readme") == "fine_tuned_from"
    assert lineage_extractor._determine_relationship("card_data") == "base_model"


@dataclass
class _FakeModelInfo:
    """
    _FakeModelInfo: Class description.
    """

    card_data: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    base_model: Optional[str] = None


class _FakeHFClient:
    """
    _FakeHFClient: Class description.
    """

    def __init__(self, info: Any, readme: str) -> None:
        """
        __init__: Function description.
        :param info:
        :param readme:
        :returns:
        """

        self._info = info
        self._readme = readme

    def get_model_info(self, hf_url: str) -> Any:
        """
        get_model_info: Function description.
        :param hf_url:
        :returns:
        """

        return self._info

    def get_model_readme(self, hf_url: str) -> str:
        """
        get_model_readme: Function description.
        :param hf_url:
        :returns:
        """

        return self._readme


def test_discover_parents_with_source_dedupes_and_limits() -> None:
    """
    test_discover_parents_with_source_dedupes_and_limits: Function description.
    :param:
    :returns:
    """

    info = _FakeModelInfo(
        card_data={"base_model": "org/base"},
        config={
            "parents": [
                "org/parent1",
                "org/parent2",
                "org/parent3",
                "org/parent4",
                "org/parent5",
                "org/parent6",
                "org/parent7",
                "org/parent8",
            ]
        },
    )
    readme = (
        "Base model: org/base\n"
        "fine-tuned from org/parent9\n"
        "fine-tuned from org/parent10\n"
        "fine-tuned from org/parent11\n"
        "fine-tuned from org/parent12\n"
    )
    hf = _FakeHFClient(info, readme)

    parents = lineage_extractor._discover_parents_with_source(
        hf, "https://huggingface.co/org/model"
    )

    assert len(parents) == lineage_extractor.MAX_PARENT_FANOUT
    slugs = [slug for slug, _source in parents]
    assert slugs == [
        "org/parent1",
        "org/parent2",
        "org/parent3",
        "org/parent4",
        "org/parent5",
        "org/parent6",
        "org/parent7",
        "org/parent8",
        "org/base",
        "org/parent9",
    ]


def test_extract_lineage_graph_builds_nodes_and_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_extract_lineage_graph_builds_nodes_and_edges: Function description.
    :param monkeypatch:
    :returns:
    """

    info = _FakeModelInfo(card_data={"base_model": "org/base"})
    hf = _FakeHFClient(info, readme="")

    class _Factory:
        """
        _Factory: Class description.
        """

        def __call__(self) -> _FakeHFClient:
            """
            __call__: Function description.
            :param:
            :returns:
            """

            return hf

    monkeypatch.setattr(lineage_extractor, "HFClient", _Factory())

    graph = lineage_extractor.extract_lineage_graph(
        artifact_id="abc123",
        source_url="https://huggingface.co/org/model",
    )

    assert graph.nodes[0].artifact_id == "abc123"
    assert graph.edges
    assert graph.edges[0].from_node_artifact_id == "abc123"


def test_extract_lineage_graph_returns_minimal_graph_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_extract_lineage_graph_returns_minimal_graph_on_failure: Function description.
    :param monkeypatch:
    :returns:
    """

    class _Boom:
        """
        _Boom: Class description.
        """

        def __init__(self) -> None:
            """
            __init__: Function description.
            :param:
            :returns:
            """

            raise RuntimeError("boom")

    monkeypatch.setattr(lineage_extractor, "HFClient", _Boom)

    graph = lineage_extractor.extract_lineage_graph(
        artifact_id="abc123",
        source_url="https://huggingface.co/org/model",
    )

    assert [node.artifact_id for node in graph.nodes] == ["abc123"]
    assert graph.edges == []


def test_extract_lineage_graph_logs_and_falls_back_when_recursive_build_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    test_extract_lineage_graph_logs_and_falls_back_when_recursive_build_fails: Function description.
    :param monkeypatch:
    :param caplog:
    :returns:
    """

    info = _FakeModelInfo(card_data={"base_model": "org/base"})
    hf = _FakeHFClient(info, readme="")

    class _Factory:
        """
        _Factory: Class description.
        """

        def __call__(self) -> _FakeHFClient:
            """
            __call__: Function description.
            :param:
            :returns:
            """

            return hf

    def boom(*args: Any, **kwargs: Any) -> None:
        """
        boom: Function description.
        :param *args:
        :param **kwargs:
        :returns:
        """

        raise RuntimeError("boom")

    monkeypatch.setattr(lineage_extractor, "HFClient", _Factory())
    monkeypatch.setattr(lineage_extractor, "_build_lineage_recursive", boom)

    with caplog.at_level("ERROR"):
        graph = lineage_extractor.extract_lineage_graph(
            artifact_id="abc123",
            source_url="https://huggingface.co/org/model",
        )

    assert [node.artifact_id for node in graph.nodes] == ["abc123"]
    assert graph.edges == []
    assert "Failed to extract lineage for abc123" in caplog.text
