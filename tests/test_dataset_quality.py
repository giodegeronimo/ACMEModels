"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for test dataset quality module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pytest

import src.metrics.dataset_quality as dataset_quality
from src.metrics.dataset_quality import DatasetQualityMetric


@dataclass
class _FakeDatasetInfo:
    """
    _FakeDatasetInfo: Class description.
    """

    card_data: Dict[str, Any]
    downloads: int = 0
    likes: int = 0
    last_modified: Optional[datetime] = None
    description: Optional[str] = None


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
        datasets: Dict[str, bool] | None = None,
        dataset_info: Dict[str, _FakeDatasetInfo] | None = None,
        model_info: Optional[_FakeModelInfo] = None,
        readme: Optional[str] = None,
        model_counts: Dict[str, int] | None = None,
    ) -> None:
        """
        __init__: Function description.
        :param datasets:
        :param dataset_info:
        :param model_info:
        :param readme:
        :param model_counts:
        :returns:
        """

        self._datasets = datasets or {}
        self._dataset_info = dataset_info or {}
        self._model_info = model_info
        self._readme = readme
        self._model_counts = model_counts or {}

    def dataset_exists(self, dataset_id: str) -> bool:
        """
        dataset_exists: Function description.
        :param dataset_id:
        :returns:
        """

        return self._datasets.get(dataset_id, False)

    def get_dataset_info(self, dataset_id: str) -> Any:
        """
        get_dataset_info: Function description.
        :param dataset_id:
        :returns:
        """

        if dataset_id not in self._dataset_info:
            raise RuntimeError("dataset info missing")
        return self._dataset_info[dataset_id]

    def get_model_info(self, repo_id: str) -> Any:
        """
        get_model_info: Function description.
        :param repo_id:
        :returns:
        """

        if self._model_info is None:
            raise RuntimeError("model info missing")
        return self._model_info

    def get_model_readme(self, repo_id: str) -> str:
        """
        get_model_readme: Function description.
        :param repo_id:
        :returns:
        """

        if self._readme is None:
            raise RuntimeError("readme missing")
        return self._readme

    def count_models_trained_on_dataset(self, dataset_id: str) -> int:
        """
        count_models_trained_on_dataset: Function description.
        :param dataset_id:
        :returns:
        """

        return self._model_counts.get(dataset_id, 0)


def _recent_datetime(days: int) -> datetime:
    """
    _recent_datetime: Function description.
    :param days:
    :returns:
    """

    return datetime.now(timezone.utc) - timedelta(days=days)


def test_quality_full_score_clamped() -> None:
    """
    test_quality_full_score_clamped: Function description.
    :param:
    :returns:
    """

    card_data = {
        "annotations_creators": ["crowdsourced"],
        "language": ["en"],
        "license": ["apache-2.0"],
        "task_categories": ["text-generation"],
        "dataset_info": {
            "splits": [
                {"name": "train", "num_examples": 20000},
            ]
        },
    }
    dataset_info = _FakeDatasetInfo(
        card_data=card_data,
        downloads=50000,
        likes=200,
        last_modified=_recent_datetime(30),
        description="Comprehensive dataset",
    )
    client = _FakeHFClient(
        datasets={"org/ds": True},
        dataset_info={"org/ds": dataset_info},
        model_info=_FakeModelInfo(
            datasets=["https://huggingface.co/datasets/org/ds"]
        ),
        model_counts={"org/ds": 120},
    )
    metric = DatasetQualityMetric(hf_client=client)

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score == pytest.approx(1.0)


def test_quality_partial_metadata() -> None:
    """
    test_quality_partial_metadata: Function description.
    :param:
    :returns:
    """

    dataset_info = _FakeDatasetInfo(
        card_data={},
        downloads=100,
        likes=5,
        last_modified=_recent_datetime(400),
        description=None,
    )
    client = _FakeHFClient(
        datasets={"org/ds": True},
        dataset_info={"org/ds": dataset_info},
        model_info=_FakeModelInfo(
            datasets=["https://huggingface.co/datasets/org/ds"]
        ),
        model_counts={"org/ds": 2},
    )
    metric = DatasetQualityMetric(hf_client=client)

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert 0.0 < score < 1.0


def test_slug_detected_from_readme() -> None:
    """
    test_slug_detected_from_readme: Function description.
    :param:
    :returns:
    """

    dataset_info = _FakeDatasetInfo(card_data={})
    client = _FakeHFClient(
        datasets={"org/ds": True},
        dataset_info={"org/ds": dataset_info},
        readme="Dataset: https://huggingface.co/datasets/org/ds",
        model_counts={"org/ds": 0},
    )
    metric = DatasetQualityMetric(hf_client=client)

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score >= 0.0
    assert client._datasets["org/ds"]


def test_quality_returns_zero_for_unknown_dataset() -> None:
    """
    test_quality_returns_zero_for_unknown_dataset: Function description.
    :param:
    :returns:
    """

    client = _FakeHFClient(datasets={}, model_counts={})
    metric = DatasetQualityMetric(hf_client=client)

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score == 0.0


def test_manifest_dataset_validation() -> None:
    """
    test_manifest_dataset_validation: Function description.
    :param:
    :returns:
    """

    dataset_info = _FakeDatasetInfo(card_data={})
    client = _FakeHFClient(
        datasets={"org/ds": True},
        dataset_info={"org/ds": dataset_info},
        model_counts={"org/ds": 0},
    )
    metric = DatasetQualityMetric(hf_client=client)

    score = metric.compute(
        {
            "hf_url": "https://huggingface.co/org/model",
            "ds_url": "https://huggingface.co/datasets/org/ds",
        }
    )

    assert isinstance(score, float)
    assert score >= 0.0


def test_quality_returns_zero_when_dataset_info_unavailable() -> None:
    """
    test_quality_returns_zero_when_dataset_info_unavailable: Function description.
    :param:
    :returns:
    """

    client = _FakeHFClient(
        datasets={"org/ds": True},
        dataset_info={},
        model_info=_FakeModelInfo(
            datasets=["https://huggingface.co/datasets/org/ds"]
        ),
        model_counts={"org/ds": 0},
    )
    metric = DatasetQualityMetric(hf_client=client)

    assert metric.compute({"hf_url": "https://huggingface.co/org/model"}) == 0.0


def test_quality_fail_stub_returns_expected_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_quality_fail_stub_returns_expected_value: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("ACME_IGNORE_FAIL", "0")
    monkeypatch.setattr(dataset_quality, "FAIL", True)

    metric = DatasetQualityMetric(hf_client=_FakeHFClient())
    score = metric.compute(
        {"hf_url": "https://huggingface.co/openai/whisper-tiny/tree/main"}
    )
    assert score == pytest.approx(0.63)


def test_dataset_reference_validation_falls_back_to_slug_slug() -> None:
    """
    test_dataset_reference_validation_falls_back_to_slug_slug: Function description.
    :param:
    :returns:
    """

    client = _FakeHFClient(datasets={"foo/foo": True})
    metric = DatasetQualityMetric(hf_client=client)

    assert metric._dataset_reference_is_valid("foo") is True


def test_dataset_score_from_manifest_helper() -> None:
    """
    test_dataset_score_from_manifest_helper: Function description.
    :param:
    :returns:
    """

    client = _FakeHFClient(datasets={"org/ds": True})
    metric = DatasetQualityMetric(hf_client=client)

    assert metric._dataset_score_from_manifest({"ds_url": ""}, None) == 0.0
    assert (
        metric._dataset_score_from_manifest(
            {"ds_url": "https://huggingface.co/datasets/org/ds"},
            "https://huggingface.co/org/model",
        )
        == 0.5
    )


def test_dataset_quality_helper_parsers_cover_front_matter() -> None:
    """
    test_dataset_quality_helper_parsers_cover_front_matter: Function description.
    :param:
    :returns:
    """

    readme = (
        "---\n"
        "datasets: [org/ds, 'org/dup']\n"
        "---\n"
        "More text https://huggingface.co/datasets/org/dup\n"
    )
    candidates = dataset_quality._extract_dataset_candidates_from_readme(readme)
    assert candidates == [
        "org/ds",
        "org/dup",
        "https://huggingface.co/datasets/org/dup",
    ]

    assert dataset_quality._extract_front_matter("nope") is None
    assert dataset_quality._extract_dataset_from_readme(readme) == (
        "https://huggingface.co/datasets/org/dup"
    )


def test_score_freshness_handles_naive_timestamps() -> None:
    """
    test_score_freshness_handles_naive_timestamps: Function description.
    :param:
    :returns:
    """

    info = _FakeDatasetInfo(
        card_data={},
        last_modified=datetime.now() - timedelta(days=30),
    )
    metric = DatasetQualityMetric(hf_client=_FakeHFClient())

    assert metric._score_freshness(info) > 0.0


def test_score_splits_awards_half_for_small_splits() -> None:
    """
    test_score_splits_awards_half_for_small_splits: Function description.
    :param:
    :returns:
    """

    info = _FakeDatasetInfo(
        card_data={"dataset_info": {"splits": [{"num_examples": 10}]}},
    )
    metric = DatasetQualityMetric(hf_client=_FakeHFClient())

    assert metric._score_splits(info) == pytest.approx(
        dataset_quality.SPLITS_WEIGHT * 0.5
    )


def test_dataset_quality_readme_fallback_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_dataset_quality_readme_fallback_toggle: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setattr(dataset_quality, "enable_readme_fallback", lambda: False)
    metric = DatasetQualityMetric(hf_client=_FakeHFClient(readme="Dataset: org/ds"))
    assert metric._slug_from_readme("https://huggingface.co/org/model") is None


def test_dataset_quality_slug_from_manifest_invalid_reference() -> None:
    """
    test_dataset_quality_slug_from_manifest_invalid_reference: Function description.
    :param:
    :returns:
    """

    metric = DatasetQualityMetric(hf_client=_FakeHFClient())
    assert (
        metric._slug_from_manifest(
            {"ds_url": "https://example.com/datasets/org/ds"},
            "https://huggingface.co/org/model",
        )
        is None
    )


def test_dataset_quality_flatten_entries_and_slug_parsing_helpers() -> None:
    """
    test_dataset_quality_flatten_entries_and_slug_parsing_helpers: Function description.
    :param:
    :returns:
    """

    assert dataset_quality._flatten_dataset_entries(None) == []
    assert dataset_quality._flatten_dataset_entries(" org/ds ") == ["org/ds"]
    assert dataset_quality._flatten_dataset_entries(["org/ds", 123, " "]) == [
        "org/ds"
    ]
    assert dataset_quality._flatten_dataset_entries({"bad": "type"}) == []

    assert dataset_quality._to_dataset_slug("") is None
    assert (
        dataset_quality._to_dataset_slug("https://example.com/datasets/org/ds")
        is None
    )
    assert (
        dataset_quality._to_dataset_slug("https://huggingface.co/datasets/org/ds")
        == "org/ds"
    )
    assert (
        dataset_quality._to_dataset_slug("https://huggingface.co/datasets/ds")
        == "ds"
    )
    assert dataset_quality._to_dataset_slug("org/ds") == "org/ds"

    assert dataset_quality._extract_dataset_from_readme("") is None


def test_dataset_quality_score_license_branches() -> None:
    """
    test_dataset_quality_score_license_branches: Function description.
    :param:
    :returns:
    """

    metric = DatasetQualityMetric(hf_client=_FakeHFClient())
    assert (
        metric._score_license(_FakeDatasetInfo(card_data={"license": "apache-2.0"}))
        == dataset_quality.LICENSE_WEIGHT
    )
    assert (
        metric._score_license(_FakeDatasetInfo(card_data={"license": ["custom"]}))
        == pytest.approx(dataset_quality.LICENSE_WEIGHT * 0.5)
    )


def test_score_freshness_old_and_midrange() -> None:
    """
    test_score_freshness_old_and_midrange: Function description.
    :param:
    :returns:
    """

    metric = DatasetQualityMetric(hf_client=_FakeHFClient())
    assert (
        metric._score_freshness(
            _FakeDatasetInfo(card_data={}, last_modified=_recent_datetime(5000))
        )
        == 0.0
    )
    assert (
        metric._score_freshness(
            _FakeDatasetInfo(card_data={}, last_modified=_recent_datetime(500))
        )
        > 0.0
    )


def test_extract_datasets_from_front_matter_multiline() -> None:
    """
    test_extract_datasets_from_front_matter_multiline: Function description.
    :param:
    :returns:
    """

    front_matter = (
        "datasets:\n"
        "  - org/ds\n"
        "  - 'org/quoted'\n"
        "other: value\n"
    )
    assert dataset_quality._extract_datasets_from_front_matter(front_matter) == [
        "org/ds",
        "org/quoted",
    ]
