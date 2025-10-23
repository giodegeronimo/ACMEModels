from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.metrics.reproducibility import ReproducibilityMetric


@dataclass
class _FakeHFClient:
    readme: str

    def get_model_readme(self, repo_id: str) -> str:
        return self.readme


def test_reproducibility_scores_out_of_box_demo() -> None:
    readme = (
        "## Quickstart\n"
        "```python\n"
        "from transformers import pipeline\n"
        "pipe = pipeline(\n"
        "    'text-generation',\n"
        "    model='gpt2',\n"
        ")\n"
        "result = pipe('Hello world')\n"
        "print(result)\n"
        "```\n"
    )
    metric = ReproducibilityMetric(
        hf_client=_FakeHFClient(readme=readme)
    )

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(1.0)


def test_reproducibility_scores_requires_debugging() -> None:
    readme = (
        "## Usage\n"
        "```python\n"
        "from transformers import pipeline\n"
        "pipe = pipeline('task', model='your-username/your-model')\n"
        "pipe('hello world')\n"
        "```\n"
    )
    metric = ReproducibilityMetric(
        hf_client=_FakeHFClient(readme=readme)
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
