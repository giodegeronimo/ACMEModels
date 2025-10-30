from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pytest

from src.metrics.reproducibility import ReproducibilityMetric


@dataclass
class _FakeHFClient:
    readme: str

    def get_model_readme(self, repo_id: str) -> str:
        return self.readme


class _FakePurdueClient:
    def __init__(self, responses: Iterable[str]) -> None:
        self._responses: List[str] = list(responses)
        self._call_count = 0

    def llm(self, prompt=None, *, messages=None, **kwargs) -> str:
        if not self._responses:
            return ""
        index = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[index]


def test_reproducibility_scores_out_of_box_demo() -> None:
    readme = (
        "## Quickstart\n"
        "```python\n"
        "from math import sqrt\n"
        "print(sqrt(4))\n"
        "```\n"
    )
    metric = ReproducibilityMetric(
        hf_client=_FakeHFClient(readme=readme)
    )

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(1.0)


def test_reproducibility_scores_llm_fix() -> None:
    readme = (
        "## Usage\n"
        "```python\n"
        "from math import sqrt\n"
        "print(sqrt('four'))\n"
        "```\n"
    )
    purdue_client = _FakePurdueClient(
        responses=[
            "```python\nfrom math import sqrt\nprint(sqrt(4))\n```"
        ]
    )
    metric = ReproducibilityMetric(
        hf_client=_FakeHFClient(readme=readme),
        purdue_client=purdue_client,
    )

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(0.5)


def test_reproducibility_scores_zero_without_demo() -> None:
    readme = (
        "# Model Card\n"
        "Install dependencies:\n"
        "```bash\n"
        "pip install transformers\n"
        "```\n"
    )
    metric = ReproducibilityMetric(
        hf_client=_FakeHFClient(readme=readme)
    )

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(0.0)


def test_reproducibility_scores_zero_when_llm_fails() -> None:
    readme = (
        "## Example\n"
        "```python\n"
        "from math import sqrt\n"
        "print(sqrt('four'))\n"
        "```\n"
    )
    purdue_client = _FakePurdueClient(responses=["", ""])
    metric = ReproducibilityMetric(
        hf_client=_FakeHFClient(readme=readme),
        purdue_client=purdue_client,
    )

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(0.0)
