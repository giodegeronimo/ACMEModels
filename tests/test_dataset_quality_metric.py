from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pytest

from src.metrics.dataset_quality import DatasetQualityMetric


@dataclass
class _FakeModelInfo:
    datasets: Any = None
    card_data: Any = None


@dataclass
class _FakeDatasetInfo:
    description: Optional[str] = "desc"
    downloads: int = 10_000
    likes: int = 100
    last_modified: Any = None
    card_data: Any = None


class _FakeHFClient:
    def __init__(
        self,
        *,
        dataset_exists: bool = True,
        dataset_info: Optional[Any] = None,
        model_info: Optional[Any] = None,
        readme: str = "",
        models_trained_on_dataset: int = 50,
    ) -> None:
        self._dataset_exists = dataset_exists
        self._dataset_info = dataset_info
        self._model_info = model_info
        self._readme = readme
        self._models_trained_on_dataset = models_trained_on_dataset

    def dataset_exists(self, dataset_id: str) -> bool:
        return self._dataset_exists

    def get_dataset_info(self, dataset_id: str) -> Any:
        if self._dataset_info is None:
            raise RuntimeError("dataset_info missing")
        return self._dataset_info

    def get_model_info(self, repo_id: str) -> Any:
        if self._model_info is None:
            raise RuntimeError("model_info missing")
        return self._model_info

    def get_model_readme(self, repo_id: str) -> str:
        return self._readme

    def count_models_trained_on_dataset(self, dataset_id: str) -> int:
        return self._models_trained_on_dataset


def test_dataset_quality_returns_zero_when_no_dataset_is_found() -> None:
    metric = DatasetQualityMetric(
        hf_client=_FakeHFClient(
            dataset_exists=False,
            dataset_info=_FakeDatasetInfo(),
            model_info=_FakeModelInfo(),
        )
    )
    assert metric.compute({"hf_url": "https://huggingface.co/acme/model"}) == 0.0


def test_dataset_quality_scores_from_manifest_dataset_url() -> None:
    now = datetime.now(timezone.utc)
    dataset_info = _FakeDatasetInfo(
        last_modified=now - timedelta(days=30),
        card_data={
            "annotations_creators": ["a"],
            "language": ["en"],
            "license": "mit",
            "task_categories": ["x"],
            "dataset_info": {"splits": [{"name": "train", "num_examples": 2000}]},
        },
    )
    metric = DatasetQualityMetric(
        hf_client=_FakeHFClient(
            dataset_exists=True,
            dataset_info=dataset_info,
            model_info=_FakeModelInfo(),
            models_trained_on_dataset=50,
        )
    )
    score = metric.compute(
        {
            "hf_url": "https://huggingface.co/acme/model",
            "ds_url": "https://huggingface.co/datasets/acme/ds",
        }
    )
    assert isinstance(score, float)
    assert 0.9 <= score <= 1.0


def test_dataset_quality_falls_back_to_metadata_when_manifest_invalid() -> None:
    now = datetime.now(timezone.utc)
    dataset_info = _FakeDatasetInfo(
        last_modified=now - timedelta(days=365),
        card_data={"license": ["unknown-license"], "dataset_info": {"splits": []}},
    )
    model_info = _FakeModelInfo(
        datasets=["https://huggingface.co/datasets/acme/ds"],
        card_data={"datasets": ["acme/ds"]},
    )
    metric = DatasetQualityMetric(
        hf_client=_FakeHFClient(
            dataset_exists=True,
            dataset_info=dataset_info,
            model_info=model_info,
            models_trained_on_dataset=0,
        )
    )
    score = metric.compute(
        {
            "hf_url": "https://huggingface.co/acme/model",
            "ds_url": "https://example.com/not-hf",
        }
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_dataset_quality_readme_fallback_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.metrics.dataset_quality.enable_readme_fallback",
        lambda: False,
    )
    metric = DatasetQualityMetric(
        hf_client=_FakeHFClient(
            dataset_exists=True,
            dataset_info=_FakeDatasetInfo(),
            model_info=_FakeModelInfo(datasets=[]),
            readme="datasets:\n  - https://huggingface.co/datasets/acme/ds\n",
        )
    )
    assert metric.compute({"hf_url": "https://huggingface.co/acme/model"}) == 0.0


def test_dataset_quality_fail_stub_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.metrics.dataset_quality.fail_stub_active",
        lambda _flag: True,
    )
    metric = DatasetQualityMetric(
        hf_client=_FakeHFClient(
            dataset_exists=False,
            dataset_info=_FakeDatasetInfo(),
            model_info=_FakeModelInfo(),
        )
    )
    score = metric.compute(
        {"hf_url": "https://huggingface.co/openai/whisper-tiny/tree/main"}
    )
    assert score == 0.63


def test_dataset_quality_returns_zero_when_dataset_info_unavailable() -> None:
    metric = DatasetQualityMetric(
        hf_client=_FakeHFClient(
            dataset_exists=True,
            dataset_info=None,
            model_info=_FakeModelInfo(),
        )
    )
    assert (
        metric.compute(
            {
                "hf_url": "https://huggingface.co/acme/model",
                "ds_url": "https://huggingface.co/datasets/acme/ds",
            }
        )
        == 0.0
    )


def test_dataset_score_from_manifest_branches() -> None:
    metric = DatasetQualityMetric(
        hf_client=_FakeHFClient(
            dataset_exists=True,
            dataset_info=_FakeDatasetInfo(),
            model_info=_FakeModelInfo(),
        )
    )
    assert metric._dataset_score_from_manifest({}, None) == 0.0
    assert (
        metric._dataset_score_from_manifest(
            {"ds_url": "https://huggingface.co/datasets/acme/ds"},
            "https://huggingface.co/acme/model",
        )
        == 0.5
    )

    invalid_metric = DatasetQualityMetric(
        hf_client=_FakeHFClient(
            dataset_exists=False,
            dataset_info=_FakeDatasetInfo(),
            model_info=_FakeModelInfo(),
        )
    )
    assert (
        invalid_metric._dataset_score_from_manifest(
            {"ds_url": "https://example.com/not-hf"},
            "https://huggingface.co/acme/model",
        )
        == 0.0
    )


def test_dataset_quality_extracts_from_readme_front_matter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.metrics.dataset_quality.enable_readme_fallback",
        lambda: True,
    )
    now = datetime.now(timezone.utc)
    dataset_info = _FakeDatasetInfo(
        last_modified=now - timedelta(days=1095),
        card_data={"license": "apache-2.0", "dataset_info": {"splits": [{"num_examples": 10}]}},
    )
    metric = DatasetQualityMetric(
        hf_client=_FakeHFClient(
            dataset_exists=True,
            dataset_info=dataset_info,
            model_info=_FakeModelInfo(datasets=[]),
            readme="---\ndatasets:\n  - https://huggingface.co/datasets/acme/ds\n---\n",
            models_trained_on_dataset=1,
        )
    )
    score = metric.compute({"hf_url": "https://huggingface.co/acme/model"})
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
