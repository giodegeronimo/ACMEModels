"""

ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from src.services import license_analysis


@dataclass
class _Info:
    """
    _Info: Class description.
    """

    license: Optional[Any] = None
    card_data: Optional[Dict[str, Any]] = None


class _HFClient:
    """
    _HFClient: Class description.
    """

    def __init__(self, *, info: Any = None, readme: str = "") -> None:
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

        if self._info is None:
            raise RuntimeError("info missing")
        return self._info

    def get_model_readme(self, repo_id: str) -> str:
        """
        get_model_readme: Function description.
        :param repo_id:
        :returns:
        """

        return self._readme


def test_collect_candidates_from_metadata_and_readme() -> None:
    """
    test_collect_candidates_from_metadata_and_readme: Function description.
    :param:
    :returns:
    """

    readme = "# Title\n## License\nThis model is MIT licensed.\n## Other\nx\n"
    info = _Info(license="Apache-2.0", card_data={"license": ["CC-BY-SA-4.0"]})
    client = _HFClient(info=info, readme=readme)

    meta, readme_candidates = license_analysis.collect_hf_license_candidates(
        client, "https://huggingface.co/org/model"
    )

    assert meta == ["Apache-2.0", "CC-BY-SA-4.0"]
    assert "MIT" in readme_candidates


def test_normalize_candidates_dedupes_and_strips() -> None:
    """
    test_normalize_candidates_dedupes_and_strips: Function description.
    :param:
    :returns:
    """

    assert license_analysis.normalize_license_candidates(
        [" MIT ", "mit", "", "Apache 2.0"]
    ) == ["MIT", "Apache-2.0"]


def test_evaluate_classification_behaviour() -> None:
    """
    test_evaluate_classification_behaviour: Function description.
    :param:
    :returns:
    """

    policy = license_analysis.load_license_policy()
    assert license_analysis.evaluate_classification(["MIT"], policy) == "compatible"
    assert (
        license_analysis.evaluate_classification(["GPL-3.0-only"], policy)
        == "incompatible"
    )
    assert (
        license_analysis.evaluate_classification(
            ["MIT", "GPL-3.0-only"], policy
        )
        == "caution"
    )
    assert license_analysis.evaluate_classification(["Unknown-License"], policy) == "unknown"


def test_load_license_policy_uses_fallback_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_load_license_policy_uses_fallback_on_exception: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setattr(license_analysis, "_load_json_list", lambda path: 1 / 0)
    policy = license_analysis.load_license_policy()
    assert policy.class_of("MIT") == "compatible"
    assert policy.class_of("GPL-3.0-only") == "incompatible"
    assert policy.class_of("OpenRAIL-M") == "caution"


def test_load_json_list_rejects_non_list(tmp_path: Path) -> None:
    """
    test_load_json_list_rejects_non_list: Function description.
    :param tmp_path:
    :returns:
    """

    path = tmp_path / "policy.json"
    path.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a list"):
        license_analysis._load_json_list(path)
