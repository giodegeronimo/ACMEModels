"""Tests for test license metric module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

from src.metrics.license import LicenseMetric


@dataclass
class _FakeModelInfo:
    license: Optional[str] = None
    card_data: Optional[Dict[str, Any]] = None


class _FakeHFClient:
    def __init__(
        self,
        *,
        model_info: Optional[Any] = None,
        readme: Optional[str] = None,
    ) -> None:
        self._model_info = model_info
        self._readme = readme

    def get_model_info(self, repo_id: str) -> Any:
        if self._model_info is None:
            raise RuntimeError("info missing")
        return self._model_info

    def get_model_readme(self, repo_id: str) -> str:
        if self._readme is None:
            raise RuntimeError("readme missing")
        return self._readme


def test_license_from_metadata_compatible_scores_full() -> None:
    info = _FakeModelInfo(license="Apache-2.0")
    metric = LicenseMetric(hf_client=_FakeHFClient(model_info=info, readme=""))

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(1.0)


def test_license_from_readme_recognized_scores_high() -> None:
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
    info = _FakeModelInfo(license="GPL-3.0-only")
    metric = LicenseMetric(hf_client=_FakeHFClient(model_info=info, readme=""))

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(0.0)


def test_license_caution_metadata_scores_mid() -> None:
    info = _FakeModelInfo(license="CC-BY-SA-4.0")
    metric = LicenseMetric(hf_client=_FakeHFClient(model_info=info, readme=""))

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(0.5)


def test_license_unknown_is_zero() -> None:
    info = _FakeModelInfo(license=None)
    metric = LicenseMetric(hf_client=_FakeHFClient(model_info=info, readme=""))

    score = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert score == pytest.approx(0.0)
