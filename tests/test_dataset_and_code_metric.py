"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for test dataset and code metric module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

import src.metrics.dataset_and_code as dataset_and_code
from src.metrics.dataset_and_code import DatasetAndCodeMetric


@dataclass
class _FakeModelInfo:
    """
    _FakeModelInfo: Class description.
    """

    datasets: Optional[list[str]] = None
    card_data: Optional[Dict[str, Any]] = None


class _FakeHFClient:
    """
    _FakeHFClient: Class description.
    """

    def __init__(
        self,
        *,
        model_info: Optional[Any] = None,
        readme: Optional[str] = None,
        dataset_exists: Optional[Dict[str, bool]] = None,
    ) -> None:
        """
        __init__: Function description.
        :param model_info:
        :param readme:
        :param dataset_exists:
        :returns:
        """

        self._model_info = model_info
        self._readme = readme
        self.info_calls: list[str] = []
        self.readme_calls: list[str] = []
        self._datasets = dataset_exists or {}

    def get_model_info(self, repo_id: str) -> Any:
        """
        get_model_info: Function description.
        :param repo_id:
        :returns:
        """

        self.info_calls.append(repo_id)
        if self._model_info is None:
            raise RuntimeError("info missing")
        return self._model_info

    def get_model_readme(self, repo_id: str) -> str:
        """
        get_model_readme: Function description.
        :param repo_id:
        :returns:
        """

        self.readme_calls.append(repo_id)
        if self._readme is None:
            raise RuntimeError("readme missing")
        return self._readme

    def dataset_exists(self, dataset_id: str) -> bool:
        """
        dataset_exists: Function description.
        :param dataset_id:
        :returns:
        """

        return self._datasets.get(dataset_id, False)


def test_score_uses_manifest_urls() -> None:
    """
    test_score_uses_manifest_urls: Function description.
    :param:
    :returns:
    """

    metric = DatasetAndCodeMetric(
        hf_client=_FakeHFClient(dataset_exists={"sample/dataset": True})
    )
    record = {
        "hf_url": "https://huggingface.co/org/model",
        "ds_url": "https://huggingface.co/datasets/sample/dataset",
        "git_url": "https://github.com/org/model",
    }

    score = metric.compute(record)

    assert score == pytest.approx(1.0)


def test_dataset_detected_from_model_info() -> None:
    """
    test_dataset_detected_from_model_info: Function description.
    :param:
    :returns:
    """

    client = _FakeHFClient(
        model_info=_FakeModelInfo(
            datasets=["https://huggingface.co/datasets/sample/dataset"]
        ),
        readme="",
        dataset_exists={"sample/dataset": True},
    )
    metric = DatasetAndCodeMetric(hf_client=client)

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(0.5)
    assert client.info_calls


def test_dataset_detected_from_readme() -> None:
    """
    test_dataset_detected_from_readme: Function description.
    :param:
    :returns:
    """

    client = _FakeHFClient(
        model_info=None,
        readme="See https://huggingface.co/datasets/org/ds for details",
        dataset_exists={"org/ds": True},
    )
    metric = DatasetAndCodeMetric(hf_client=client)

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(0.5)
    assert client.readme_calls


def test_code_detected_from_readme() -> None:
    """
    test_code_detected_from_readme: Function description.
    :param:
    :returns:
    """

    client = _FakeHFClient(
        model_info=_FakeModelInfo(),
        readme="Source: https://github.com/org/project",
        dataset_exists={},
    )
    metric = DatasetAndCodeMetric(hf_client=client)

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(0.5)
    assert client.readme_calls


def test_dataset_metadata_requires_link() -> None:
    """
    test_dataset_metadata_requires_link: Function description.
    :param:
    :returns:
    """

    client = _FakeHFClient(
        model_info=_FakeModelInfo(datasets=["bookcorpus"]),
        readme="",
        dataset_exists={},
    )
    metric = DatasetAndCodeMetric(hf_client=client)

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(0.0)


def test_dataset_and_code_fail_stub_returns_expected_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_dataset_and_code_fail_stub_returns_expected_value: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("ACME_IGNORE_FAIL", "0")
    monkeypatch.setattr(dataset_and_code, "FAIL", True)

    metric = DatasetAndCodeMetric(hf_client=_FakeHFClient())
    score = metric.compute(
        {"hf_url": "https://huggingface.co/openai/whisper-tiny/tree/main"}
    )
    assert score == pytest.approx(1.0)


def test_dataset_reference_validation_falls_back_to_slug_slug() -> None:
    """
    test_dataset_reference_validation_falls_back_to_slug_slug: Function description.
    :param:
    :returns:
    """

    metric = DatasetAndCodeMetric(
        hf_client=_FakeHFClient(dataset_exists={"foo/foo": True})
    )
    assert metric._dataset_reference_is_valid("foo") is True


def test_readme_fallback_disabled_skips_readme_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_readme_fallback_disabled_skips_readme_calls: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("ACME_ENABLE_README_FALLBACK", "0")
    client = _FakeHFClient(
        model_info=_FakeModelInfo(),
        readme="See https://huggingface.co/datasets/org/ds",
        dataset_exists={"org/ds": True},
    )
    metric = DatasetAndCodeMetric(hf_client=client)

    assert metric.compute({"hf_url": "https://huggingface.co/org/model"}) == 0.0
    assert client.readme_calls == []


def test_collect_dataset_urls_considers_card_data() -> None:
    """
    test_collect_dataset_urls_considers_card_data: Function description.
    :param:
    :returns:
    """

    client = _FakeHFClient(
        model_info=_FakeModelInfo(
            datasets=None,
            card_data={"datasets": ["https://huggingface.co/datasets/org/ds"]},
        ),
        readme="",
        dataset_exists={"org/ds": True},
    )
    metric = DatasetAndCodeMetric(hf_client=client)

    assert metric.compute({"hf_url": "https://huggingface.co/org/model"}) == (
        pytest.approx(0.5)
    )


def test_dataset_and_code_helper_parsers_cover_front_matter() -> None:
    """
    test_dataset_and_code_helper_parsers_cover_front_matter: Function description.
    :param:
    :returns:
    """

    readme = (
        "---\n"
        "datasets:\n"
        "  - org/ds\n"
        "---\n"
        "Body https://huggingface.co/datasets/org/ds\n"
    )
    candidates = dataset_and_code._extract_dataset_candidates_from_readme(readme)
    assert candidates == ["org/ds", "https://huggingface.co/datasets/org/ds"]

    assert dataset_and_code._extract_front_matter("no front") is None
    assert dataset_and_code._flatten_dataset_entries({"nope": True}) == []
    assert (
        dataset_and_code._to_dataset_slug("https://example.com/datasets/x/y")
        is None
    )
