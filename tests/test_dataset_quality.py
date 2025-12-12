"""Tests for test dataset quality module."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pytest

from src.metrics.dataset_quality import DatasetQualityMetric


@dataclass
class _FakeDatasetInfo:
    card_data: Dict[str, Any]
    downloads: int = 0
    likes: int = 0
    last_modified: Optional[datetime] = None
    description: Optional[str] = None


@dataclass
class _FakeModelInfo:
    datasets: Optional[list[str]] = None
    card_data: Optional[Dict[str, Any]] = None


class _FakeHFClient:
    def __init__(
        self,
        *,
        datasets: Dict[str, bool] | None = None,
        dataset_info: Dict[str, _FakeDatasetInfo] | None = None,
        model_info: Optional[_FakeModelInfo] = None,
        readme: Optional[str] = None,
        model_counts: Dict[str, int] | None = None,
    ) -> None:
        self._datasets = datasets or {}
        self._dataset_info = dataset_info or {}
        self._model_info = model_info
        self._readme = readme
        self._model_counts = model_counts or {}

    def dataset_exists(self, dataset_id: str) -> bool:
        return self._datasets.get(dataset_id, False)

    def get_dataset_info(self, dataset_id: str) -> Any:
        if dataset_id not in self._dataset_info:
            raise RuntimeError("dataset info missing")
        return self._dataset_info[dataset_id]

    def get_model_info(self, repo_id: str) -> Any:
        if self._model_info is None:
            raise RuntimeError("model info missing")
        return self._model_info

    def get_model_readme(self, repo_id: str) -> str:
        if self._readme is None:
            raise RuntimeError("readme missing")
        return self._readme

    def count_models_trained_on_dataset(self, dataset_id: str) -> int:
        return self._model_counts.get(dataset_id, 0)


def _recent_datetime(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


def test_quality_full_score_clamped() -> None:
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
    client = _FakeHFClient(datasets={}, model_counts={})
    metric = DatasetQualityMetric(hf_client=client)

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score == 0.0


def test_manifest_dataset_validation() -> None:
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
