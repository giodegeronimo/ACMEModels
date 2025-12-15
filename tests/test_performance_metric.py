"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for test performance metric module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pytest

import src.metrics.performance as performance
from src.metrics.performance import PerformanceMetric


@dataclass
class _FakeModelInfo:
    """
    _FakeModelInfo: Class description.
    """

    card_data: Dict[str, Any]


class _FakeHFClient:
    """
    _FakeHFClient: Class description.
    """

    def __init__(
        self,
        *,
        info: Optional[_FakeModelInfo] = None,
        readme: str = "",
    ) -> None:
        """
        __init__: Function description.
        :param info:
        :param readme:
        :returns:
        """

        self._info = info
        self._readme = readme

    def get_model_info(self, repo_id: str) -> Any:
        """
        get_model_info: Function description.
        :param repo_id:
        :returns:
        """

        return self._info

    def get_model_readme(self, repo_id: str) -> str:
        """
        get_model_readme: Function description.
        :param repo_id:
        :returns:
        """

        return self._readme


class _FakePurdueClient:
    """
    _FakePurdueClient: Class description.
    """

    def __init__(self, responses: Optional[List[str]] = None) -> None:
        """
        __init__: Function description.
        :param responses:
        :returns:
        """

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
        """
        llm: Function description.
        :param prompt:
        :param messages:
        :param model:
        :param stream:
        :param temperature:
        :param **extra:
        :returns:
        """

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
    """
    _metric: Function description.
    :param info:
    :param readme:
    :param purdue:
    :returns:
    """

    return PerformanceMetric(
        hf_client=_FakeHFClient(info=info, readme=readme),
        purdue_client=purdue,
    )


def test_performance_detects_structured_metadata() -> None:
    """
    test_performance_detects_structured_metadata: Function description.
    :param:
    :returns:
    """

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
    """
    test_performance_llm_positive: Function description.
    :param:
    :returns:
    """

    readme = "## Results\nThis model achieves strong results on MNLI."
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


def test_performance_readme_regex_positive_without_llm() -> None:
    """
    test_performance_readme_regex_positive_without_llm: Function description.
    :param:
    :returns:
    """

    readme = "## Results\nThis model achieves 92% accuracy on MNLI."
    metric = _metric(None, readme, purdue=None)

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score == pytest.approx(1.0)


def test_performance_llm_negative() -> None:
    """
    test_performance_llm_negative: Function description.
    :param:
    :returns:
    """

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
    """
    test_performance_empty_readme_returns_zero: Function description.
    :param:
    :returns:
    """

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
    """
    test_performance_fallback_disabled: Function description.
    :param monkeypatch:
    :returns:
    """

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


def test_performance_returns_zero_without_hf_url() -> None:
    """
    test_performance_returns_zero_without_hf_url: Function description.
    :param:
    :returns:
    """

    metric = _metric(None, readme="", purdue=None)

    assert metric.compute({"hf_url": ""}) == pytest.approx(0.0)


def test_performance_fail_stub_returns_known_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_performance_fail_stub_returns_known_score: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("ACME_IGNORE_FAIL", "0")
    monkeypatch.setattr(performance, "FAIL", True)

    metric = _metric(None, readme="", purdue=None)
    score = metric.compute(
        {"hf_url": "https://huggingface.co/parvk11/audience_classifier_model"}
    )

    assert score == pytest.approx(1.0)


def test_performance_llm_extraction_without_yes_no_returns_false() -> None:
    """
    test_performance_llm_extraction_without_yes_no_returns_false: Function description.
    :param:
    :returns:
    """

    readme = "## Results\nNo explicit numbers here."
    purdue = _FakePurdueClient(
        [
            "Analysis text...",
            "maybe",
        ]
    )
    metric = _metric(None, readme, purdue=purdue)

    assert metric.compute({"hf_url": "https://huggingface.co/org/model"}) == (
        pytest.approx(0.0)
    )


def test_has_structured_claims_accepts_multiple_schema_variants() -> None:
    """
    test_has_structured_claims_accepts_multiple_schema_variants: Function description.
    :param:
    :returns:
    """

    class Info:
        """
        Info: Class description.
        """

        cardData = {"benchmark": "glue"}  # noqa: N815 - upstream schema

    assert performance._has_structured_claims(Info()) is True


def test_readme_numeric_claims_heuristics() -> None:
    """
    test_readme_numeric_claims_heuristics: Function description.
    :param:
    :returns:
    """

    assert performance._readme_has_numeric_claims("") is False
    assert performance._readme_has_numeric_claims("no digits") is False
    assert performance._readme_has_numeric_claims("F1: 89.5") is True
    assert (
        performance._readme_has_numeric_claims("This was released in 2024.")
        is False
    )
    assert performance._readme_has_numeric_claims("Perplexity: 12.3") is True
    assert performance._readme_has_numeric_claims("ppl: 0") is False

    assert (
        performance._readme_has_numeric_claims("```\\nAccuracy: 99\\n```")
        is False
    )


def test_performance_handles_hf_client_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_performance_handles_hf_client_exceptions: Function description.
    :param monkeypatch:
    :returns:
    """

    class FailingHFClient(_FakeHFClient):
        """
        FailingHFClient: Class description.
        """

        def get_model_info(self, repo_id: str) -> Any:
            """
            get_model_info: Function description.
            :param repo_id:
            :returns:
            """

            raise RuntimeError("nope")

        def get_model_readme(self, repo_id: str) -> str:
            """
            get_model_readme: Function description.
            :param repo_id:
            :returns:
            """

            raise RuntimeError("nope")

    monkeypatch.setenv("ACME_ENABLE_README_FALLBACK", "1")
    metric = PerformanceMetric(hf_client=FailingHFClient())

    assert metric.compute({"hf_url": "https://huggingface.co/org/model"}) == (
        pytest.approx(0.0)
    )


def test_performance_handles_missing_purdue_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_performance_handles_missing_purdue_client: Function description.
    :param monkeypatch:
    :returns:
    """

    import src.clients.purdue_client as purdue_client

    def failing_constructor() -> Any:
        """
        failing_constructor: Function description.
        :param:
        :returns:
        """

        raise RuntimeError("boom")

    monkeypatch.setattr(purdue_client, "PurdueClient", failing_constructor)
    monkeypatch.setenv("ACME_ENABLE_README_FALLBACK", "1")
    metric = _metric(None, "No metrics here.", purdue=None)

    assert metric.compute({"hf_url": "https://huggingface.co/org/model"}) == (
        pytest.approx(0.0)
    )


def test_performance_handles_llm_failures() -> None:
    """
    test_performance_handles_llm_failures: Function description.
    :param:
    :returns:
    """

    purdue = _FakePurdueClient([])
    metric = _metric(None, "No metrics here.", purdue=purdue)

    assert metric.compute({"hf_url": "https://huggingface.co/org/model"}) == (
        pytest.approx(0.0)
    )
