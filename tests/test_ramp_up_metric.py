"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for test ramp up metric module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pytest

from src.metrics.ramp_up import RampUpMetric


@dataclass
class _FakeSibling:
    """
    _FakeSibling: Class description.
    """

    rfilename: str


@dataclass
class _FakeModelInfo:
    """
    _FakeModelInfo: Class description.
    """

    siblings: list[_FakeSibling]
    pipeline_tag: Optional[str] = None
    library_name: Optional[str] = None


class _FakeHFClient:
    """
    _FakeHFClient: Class description.
    """

    def __init__(
        self,
        *,
        model_info: Optional[_FakeModelInfo] = None,
        readme: Optional[str] = None,
    ) -> None:
        """
        __init__: Function description.
        :param model_info:
        :param readme:
        :returns:
        """

        self._model_info = model_info
        self._readme = readme

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


class _FakePurdueClient:
    """
    _FakePurdueClient: Class description.
    """

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        *,
        fail: bool = False,
    ) -> None:
        """
        __init__: Function description.
        :param responses:
        :param fail:
        :returns:
        """

        self._responses = responses or []
        self._fail = fail
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
        if self._fail:
            raise RuntimeError("LLM failure")
        if not self._responses:
            raise RuntimeError("No responses configured")
        return self._responses.pop(0)


def test_ramp_up_full_score_is_clamped() -> None:
    """
    test_ramp_up_full_score_is_clamped: Function description.
    :param:
    :returns:
    """

    readme = (
        "# Model Title\n"
        "## Quickstart\n"
        "Use the following code:\n"
        "This section explains in detail how to prepare the environment.\n"
        "It includes prerequisites, step-by-step instructions, and tips.\n"
        "```python\n"
        "from transformers import pipeline\n"
        "pipeline('task', model='org/model')\n"
        "```\n"
        "```bash\n"
        "pip install transformers\n"
        "```\n"
        "## Installation\n"
        "Follow these steps to install the package and its dependencies.\n"
        "## Advanced Usage\n"
        "More advanced examples and extended commentary.\n"
        "[Docs](https://huggingface.co/docs)\n"
        "[Demo](https://huggingface.co/spaces/example)\n"
        "[Blog](https://medium.com/example)\n"
    )
    model_info = _FakeModelInfo(
        siblings=[
            _FakeSibling("README.md"),
            _FakeSibling("demo_notebook.ipynb"),
            _FakeSibling("usage_example.py"),
        ],
        pipeline_tag="text-classification",
        library_name="transformers",
    )
    purdue = _FakePurdueClient(
        responses=[
            "Reasoning... Final rating: 0.92",
            "0.92",
        ]
    )
    metric = RampUpMetric(
        hf_client=_FakeHFClient(model_info=model_info, readme=readme),
        purdue_client=purdue,
    )

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score == pytest.approx(0.97, rel=1e-2)


def test_ramp_up_no_readme_yields_zero() -> None:
    """
    test_ramp_up_no_readme_yields_zero: Function description.
    :param:
    :returns:
    """

    metric = RampUpMetric(
        hf_client=_FakeHFClient(readme=""),
        purdue_client=_FakePurdueClient(fail=True),
    )

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score == pytest.approx(0.0)


def test_ramp_up_partial_scores_accumulate() -> None:
    """
    test_ramp_up_partial_scores_accumulate: Function description.
    :param:
    :returns:
    """

    readme = (
        "Short card\n"
        "## Usage\n"
        "```python\n"
        "print('hello')\n"
        "```\n"
    )
    model_info = _FakeModelInfo(
        siblings=[_FakeSibling("README.md")],
        pipeline_tag=None,
        library_name=None,
    )
    metric = RampUpMetric(
        hf_client=_FakeHFClient(model_info=model_info, readme=readme),
        purdue_client=_FakePurdueClient(fail=True),
    )

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert 0.0 < score < 1.0


def test_ramp_up_handles_missing_purdue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_ramp_up_handles_missing_purdue: Function description.
    :param monkeypatch:
    :returns:
    """

    def fail_client() -> None:
        """
        fail_client: Function description.
        :param:
        :returns:
        """

        raise RuntimeError("GEN_AI_STUDIO_API_KEY missing")

    monkeypatch.setattr(
        "src.metrics.ramp_up.PurdueClient",
        fail_client,
    )

    long_readme = "Documentation\n" + ("content\n" * 1000)
    metric = RampUpMetric(
        hf_client=_FakeHFClient(
            model_info=_FakeModelInfo(siblings=[]),
            readme=long_readme,
        )
    )

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert isinstance(score, float)
    assert score >= 0.0
