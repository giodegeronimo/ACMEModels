"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for test license metric module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

import src.metrics.license as license_metric
from src.metrics.license import LicenseMetric


@dataclass
class _FakeModelInfo:
    """
    _FakeModelInfo: Class description.
    """

    license: Optional[str] = None
    card_data: Optional[Dict[str, Any]] = None


class _FakeHFClient:
    """
    _FakeHFClient: Class description.
    """

    def __init__(
        self,
        *,
        model_info: Optional[Any] = None,
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
            raise RuntimeError("info missing")
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


def test_license_from_metadata_compatible_scores_full() -> None:
    """
    test_license_from_metadata_compatible_scores_full: Function description.
    :param:
    :returns:
    """

    info = _FakeModelInfo(license="Apache-2.0")
    metric = LicenseMetric(hf_client=_FakeHFClient(model_info=info, readme=""))

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(1.0)


def test_license_from_readme_recognized_scores_high() -> None:
    """
    test_license_from_readme_recognized_scores_high: Function description.
    :param:
    :returns:
    """

    readme = """
    # Title
    ## License
    This project is released under the MIT License.
    """
    metric = LicenseMetric(
        hf_client=_FakeHFClient(model_info=_FakeModelInfo(), readme=readme)
    )

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(1.0)


def test_license_incompatible_metadata_scores_low() -> None:
    """
    test_license_incompatible_metadata_scores_low: Function description.
    :param:
    :returns:
    """

    info = _FakeModelInfo(license="GPL-3.0-only")
    metric = LicenseMetric(hf_client=_FakeHFClient(model_info=info, readme=""))

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(0.0)


def test_license_caution_metadata_scores_mid() -> None:
    """
    test_license_caution_metadata_scores_mid: Function description.
    :param:
    :returns:
    """

    info = _FakeModelInfo(license="CC-BY-SA-4.0")
    metric = LicenseMetric(hf_client=_FakeHFClient(model_info=info, readme=""))

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(0.5)


def test_license_unknown_is_zero() -> None:
    """
    test_license_unknown_is_zero: Function description.
    :param:
    :returns:
    """

    info = _FakeModelInfo(license=None)
    metric = LicenseMetric(hf_client=_FakeHFClient(model_info=info, readme=""))

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(0.0)


def test_license_returns_zero_when_hf_url_missing() -> None:
    """
    test_license_returns_zero_when_hf_url_missing: Function description.
    :param:
    :returns:
    """

    metric = LicenseMetric(hf_client=_FakeHFClient(model_info=_FakeModelInfo(), readme=""))
    assert metric.compute({}) == 0.0


def test_license_fail_stub_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_license_fail_stub_path: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("ACME_IGNORE_FAIL", "0")
    monkeypatch.setattr(license_metric, "FAIL", True)
    monkeypatch.setattr(license_metric, "fail_stub_active", lambda flag: True)
    metric = LicenseMetric(hf_client=_FakeHFClient(model_info=_FakeModelInfo(), readme=""))

    score = metric.compute(
        {"hf_url": "https://huggingface.co/openai/whisper-tiny/tree/main"}
    )
    assert score == pytest.approx(0.9)


def test_license_unknown_classification_uses_else_branch() -> None:
    """
    test_license_unknown_classification_uses_else_branch: Function description.
    :param:
    :returns:
    """

    info = _FakeModelInfo(license="Weird License", card_data={})
    metric = LicenseMetric(hf_client=_FakeHFClient(model_info=info, readme=""))

    assert metric.compute({"hf_url": "https://huggingface.co/org/model"}) == 0.0
