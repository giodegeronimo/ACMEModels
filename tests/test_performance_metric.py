from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

from src.metrics.performance import PerformanceMetric


@dataclass
class _FakeModelInfo:
    card_data: Dict[str, Any]
    downloads: int = 0
    likes: int = 0
    lastModified: Optional[str] = None


class _FakeHFClient:
    def __init__(
        self,
        *,
        info: Optional[_FakeModelInfo] = None,
        readme: str = "",
    ) -> None:
        self._info = info
        self._readme = readme

    def get_model_info(self, repo_id: str) -> Any:
        if self._info is None:
            raise RuntimeError("info missing")
        return self._info

    def get_model_readme(self, repo_id: str) -> str:
        return self._readme


def _metric(info: Optional[_FakeModelInfo], readme: str) -> PerformanceMetric:
    return PerformanceMetric(hf_client=_FakeHFClient(info=info, readme=readme))


def test_performance_full_metadata_score() -> None:
    info = _FakeModelInfo(
        card_data={
            "model-index": [
                {
                    "results": [
                        {
                            "dataset": {"name": "SQuAD"},
                            "metrics": [
                                {"type": "f1", "value": 89.5},
                                {"type": "exact_match", "value": 82.1},
                            ],
                        }
                    ]
                }
            ]
        },
        downloads=250000,
        likes=900,
        lastModified="2024-08-01T10:00:00Z",
    )
    metric = _metric(info, readme="")

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score == pytest.approx(1.0)


def test_performance_readme_table() -> None:
    readme = (
        "## Results\n"
        "| Dataset | Metric | Score |\n"
        "| --- | --- | --- |\n"
        "| MNLI | accuracy | 87.5 |\n"
    )
    metric = _metric(None, readme)

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score > 0.5


def test_performance_no_claims() -> None:
    metric = _metric(None, "")

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score == pytest.approx(0.0)


def test_performance_repro_signals() -> None:
    info = _FakeModelInfo(
        card_data={
            "paper": "https://arxiv.org/abs/1234.5678",
            "eval": ["python eval.py"],
        },
        downloads=1500,
        likes=50,
        lastModified="2024-05-01T00:00:00Z",
    )
    readme = "Run evaluation: python eval.py"
    metric = _metric(info, readme)

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score > 0.3
