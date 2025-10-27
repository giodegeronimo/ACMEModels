"""Tests for test performance metric module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pytest

from src.metrics.performance import PerformanceMetric


@dataclass
class _FakeModelInfo:
    card_data: Dict[str, Any]


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
        return self._info

    def get_model_readme(self, repo_id: str) -> str:
        return self._readme


class _FakePurdueClient:
    def __init__(self, responses: Optional[List[str]] = None) -> None:
        self._responses = list(responses or [])
        self.calls: List[str] = []

    def llm(
        self,
        prompt: Optional[str] = None,
        *,
        messages: Optional[Sequence[Dict[str, str]]] = None,
        model: str = "llama3.1:latest",
        stream: bool = False,
        temperature: float = 0.0,
        **extra: Any,
    ) -> str:
        if messages is not None:
            self.calls.append(messages[-1]["content"])
        else:
            self.calls.append(prompt or "")
        if not self._responses:
            raise RuntimeError("No responses configured")
        return self._responses.pop(0)


def _metric(
    info: Optional[_FakeModelInfo],
    readme: str,
    purdue: Optional[_FakePurdueClient] = None,
) -> PerformanceMetric:
    return PerformanceMetric(
        hf_client=_FakeHFClient(info=info, readme=readme),
        purdue_client=purdue,
    )


def test_performance_detects_structured_metadata() -> None:
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
        }
    )
    purdue = _FakePurdueClient()
    metric = _metric(info, readme="", purdue=purdue)

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score == pytest.approx(1.0)
    assert purdue.calls == []


def test_performance_llm_positive() -> None:
    readme = "## Results\nThis model achieves 92% accuracy on MNLI."
    purdue = _FakePurdueClient(
        [
            "Analysis... Final answer: YES",
            "YES",
        ]
    )
    metric = _metric(None, readme, purdue=purdue)

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score == pytest.approx(1.0)
    assert len(purdue.calls) == 2


def test_performance_llm_negative() -> None:
    readme = "## Model Card\nNo evaluation results provided."
    purdue = _FakePurdueClient(
        [
            "Analysis... Final answer: NO",
            "NO",
        ]
    )
    metric = _metric(None, readme, purdue=purdue)

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score == pytest.approx(0.0)


def test_performance_empty_readme_returns_zero() -> None:
    purdue = _FakePurdueClient([
        "Analysis... Final answer: YES",
        "YES",
    ])
    metric = _metric(None, "\n", purdue=purdue)

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score == pytest.approx(0.0)
    assert purdue.calls == []


def test_performance_fallback_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACME_ENABLE_README_FALLBACK", "0")
    purdue = _FakePurdueClient(
        [
            "Analysis... Final answer: YES",
            "YES",
        ]
    )
    metric = _metric(None, "Results...", purdue=purdue)

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score == pytest.approx(0.0)
    assert purdue.calls == []
