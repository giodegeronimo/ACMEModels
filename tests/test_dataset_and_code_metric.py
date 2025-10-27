"""Tests for test dataset and code metric module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

from src.metrics.dataset_and_code import DatasetAndCodeMetric


@dataclass
class _FakeModelInfo:
    datasets: Optional[list[str]] = None
    card_data: Optional[Dict[str, Any]] = None


class _FakeHFClient:
    def __init__(
        self,
        *,
        model_info: Optional[Any] = None,
        readme: Optional[str] = None,
        dataset_exists: Optional[Dict[str, bool]] = None,
    ) -> None:
        self._model_info = model_info
        self._readme = readme
        self.info_calls: list[str] = []
        self.readme_calls: list[str] = []
        self._datasets = dataset_exists or {}

    def get_model_info(self, repo_id: str) -> Any:
        self.info_calls.append(repo_id)
        if self._model_info is None:
            raise RuntimeError("info missing")
        return self._model_info

    def get_model_readme(self, repo_id: str) -> str:
        self.readme_calls.append(repo_id)
        if self._readme is None:
            raise RuntimeError("readme missing")
        return self._readme

    def dataset_exists(self, dataset_id: str) -> bool:
        return self._datasets.get(dataset_id, False)


def test_score_uses_manifest_urls() -> None:
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
    client = _FakeHFClient(
        model_info=_FakeModelInfo(datasets=["bookcorpus"]),
        readme="",
        dataset_exists={},
    )
    metric = DatasetAndCodeMetric(hf_client=client)

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(0.0)
