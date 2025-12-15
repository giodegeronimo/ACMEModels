"""

ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pytest

import src.metrics.reproducibility as reproducibility
from src.metrics.reproducibility import ReproducibilityMetric


@dataclass
class _FakeHFClient:
    """
    _FakeHFClient: Class description.
    """

    readme: str

    def get_model_readme(self, repo_id: str) -> str:
        """
        get_model_readme: Function description.
        :param repo_id:
        :returns:
        """

        return self.readme


class _FakePurdueClient:
    """
    _FakePurdueClient: Class description.
    """

    def __init__(self, responses: Iterable[str]) -> None:
        """
        __init__: Function description.
        :param responses:
        :returns:
        """

        self._responses: List[str] = list(responses)
        self._call_count = 0

    def llm(self, prompt=None, *, messages=None, **kwargs) -> str:
        """
        llm: Function description.
        :param prompt:
        :param messages:
        :param **kwargs:
        :returns:
        """

        if not self._responses:
            return ""
        index = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[index]


def test_reproducibility_scores_out_of_box_demo() -> None:
    """
    test_reproducibility_scores_out_of_box_demo: Function description.
    :param:
    :returns:
    """

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
    """
    test_reproducibility_scores_llm_fix: Function description.
    :param:
    :returns:
    """

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
    """
    test_reproducibility_scores_zero_without_demo: Function description.
    :param:
    :returns:
    """

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
    """
    test_reproducibility_scores_zero_when_llm_fails: Function description.
    :param:
    :returns:
    """

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


def test_reproducibility_fail_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_reproducibility_fail_stub: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("ACME_IGNORE_FAIL", "0")
    monkeypatch.setattr(reproducibility, "FAIL", True)
    metric = ReproducibilityMetric(hf_client=_FakeHFClient(readme=""))

    score = metric.compute(
        {"hf_url": "https://huggingface.co/parvk11/audience_classifier_model"}
    )
    assert score == pytest.approx(0.0)


def test_reproducibility_returns_zero_without_hf_url() -> None:
    """
    test_reproducibility_returns_zero_without_hf_url: Function description.
    :param:
    :returns:
    """

    metric = ReproducibilityMetric(hf_client=_FakeHFClient(readme=""))
    assert metric.compute({"hf_url": ""}) == pytest.approx(0.0)


def test_run_code_block_handles_empty_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_run_code_block_handles_empty_and_timeout: Function description.
    :param monkeypatch:
    :returns:
    """

    metric = ReproducibilityMetric(hf_client=_FakeHFClient(readme=""))

    ok, error = metric._run_code_block("  \n")
    assert ok is False
    assert "empty after dedent" in error

    def raise_timeout(*args, **kwargs):  # type: ignore[no-untyped-def]
        """
        raise_timeout: Function description.
        :param *args:
        :param **kwargs:
        :returns:
        """

        raise reproducibility.subprocess.TimeoutExpired(cmd=args[0], timeout=1)

    monkeypatch.setattr(reproducibility.subprocess, "run", raise_timeout)
    ok, error = metric._run_code_block("print('x')")
    assert ok is False
    assert "timed out" in error


def test_run_code_block_combines_stdout_and_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_run_code_block_combines_stdout_and_stderr: Function description.
    :param monkeypatch:
    :returns:
    """

    metric = ReproducibilityMetric(hf_client=_FakeHFClient(readme=""))

    class _Completed:
        """
        _Completed: Class description.
        """

        returncode = 1
        stdout = "out"
        stderr = "err"

    monkeypatch.setattr(reproducibility.subprocess, "run", lambda *_, **__: _Completed())
    ok, error = metric._run_code_block("print('x')")
    assert ok is False
    assert error == "out\nerr"


def test_attempt_llm_fix_short_circuits_without_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_attempt_llm_fix_short_circuits_without_client: Function description.
    :param monkeypatch:
    :returns:
    """

    metric = ReproducibilityMetric(hf_client=_FakeHFClient(readme=""))
    monkeypatch.setattr(metric, "_get_purdue_client", lambda: None)

    ok, score = metric._attempt_llm_fix("print('x')", "boom")
    assert ok is False
    assert score is None


def test_get_purdue_client_handles_construction_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_get_purdue_client_handles_construction_errors: Function description.
    :param monkeypatch:
    :returns:
    """

    import src.clients.purdue_client as purdue_client

    monkeypatch.setattr(purdue_client, "PurdueClient", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    metric = ReproducibilityMetric(hf_client=_FakeHFClient(readme=""))
    assert metric._get_purdue_client() is None


def test_score_readme_handles_placeholders_and_llm_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_score_readme_handles_placeholders_and_llm_scores: Function description.
    :param monkeypatch:
    :returns:
    """

    metric = ReproducibilityMetric(hf_client=_FakeHFClient(readme=""))
    monkeypatch.setattr(metric, "_run_code_block", lambda _: (False, "boom"))

    readme_with_placeholder = "```python\nimport os\nprint('<MODEL_ID>')\n```"
    assert metric._score_readme(readme_with_placeholder) == pytest.approx(1.0)

    monkeypatch.setattr(metric, "_has_placeholders", lambda _: False)
    monkeypatch.setattr(metric, "_attempt_llm_fix", lambda *_: (True, 1.0))
    assert metric._score_readme("```python\nimport os\nprint('x')\n```") == (
        pytest.approx(1.0)
    )


def test_misc_reproducibility_helpers_cover_branches() -> None:
    """
    test_misc_reproducibility_helpers_cover_branches: Function description.
    :param:
    :returns:
    """

    metric = ReproducibilityMetric(hf_client=_FakeHFClient(readme=""))

    assert metric._score_readme("no code blocks") == pytest.approx(0.0)
    assert metric._looks_like_demo("") is False
    assert metric._looks_like_demo("# comment only") is False
    assert metric._looks_like_demo("pipeline('task')") is True
    assert metric._looks_like_demo("model = 'abc'") is True
    assert metric._looks_like_demo(">>> from math import sqrt") is True

    assert (
        metric._extract_llm_score("print('x')\n\n# SCORE: 1.0\n\n")
        == pytest.approx(1.0)
    )
