from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.services import license_analysis


def test_split_and_normalize_license_candidates() -> None:
    parts = license_analysis._split_license_field("MIT OR Apache-2.0")
    assert parts == ["MIT", "Apache-2.0"]
    normalized = license_analysis.normalize_license_candidates(parts)
    assert normalized == ["MIT", "Apache-2.0"]

    parts = license_analysis._split_license_field(["MIT", ["Apache 2.0"]])
    assert "MIT" in parts
    assert "Apache 2.0" in parts


def test_load_license_policy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        license_analysis,
        "_load_json_list",
        lambda _p: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    policy = license_analysis.load_license_policy()
    assert policy.class_of("MIT") == "compatible"
    assert policy.class_of("GPL-3.0-only") == "incompatible"
    assert policy.class_of("Custom") == "caution"
    assert "MIT" in policy.all_slugs


def test_load_license_policy_from_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_loader(path: Path) -> list[str]:
        name = path.name
        if "incompatible" in name:
            return ["GPL-3.0-only"]
        if "caution" in name:
            return ["Custom"]
        if "compatible" in name:
            return ["MIT"]
        raise AssertionError(f"unexpected policy file name: {name}")

    monkeypatch.setattr(license_analysis, "_load_json_list", fake_loader)
    policy = license_analysis.load_license_policy()
    assert policy.class_of("mit") == "compatible"
    assert policy.class_of("gpl-3.0-only") == "incompatible"
    assert policy.class_of("custom") == "caution"


def test_evaluate_classification_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        license_analysis,
        "_load_json_list",
        lambda _p: ["MIT"] if "compatible" in _p.name else [],
    )
    policy = license_analysis.load_license_policy()
    assert license_analysis.evaluate_classification([], policy) == "unknown"
    assert license_analysis.evaluate_classification(["MIT"], policy) == "compatible"
    assert license_analysis.evaluate_classification(["Unknown"], policy) == "unknown"

    policy2 = license_analysis._LicensePolicy(
        compatible={"mit": "MIT"},
        caution={"custom": "Custom"},
        incompatible={"gpl-3.0-only": "GPL-3.0-only"},
        all_slugs={"MIT", "Custom", "GPL-3.0-only"},
    )
    assert license_analysis.evaluate_classification(["GPL-3.0-only"], policy2) == "incompatible"
    assert license_analysis.evaluate_classification(["MIT", "GPL-3.0-only"], policy2) == "caution"
    assert license_analysis.evaluate_classification(["Custom"], policy2) == "caution"


def test_collect_hf_license_candidates_from_metadata_and_readme() -> None:
    class FakeInfo:
        license = "MIT"
        card_data = {"license": ["Apache-2.0"]}

    class FakeClient:
        def get_model_info(self, hf_url: str) -> Any:
            return FakeInfo()

        def get_model_readme(self, hf_url: str) -> str:
            return "# License\nThis project is released under GPL-3.0.\n"

    meta, readme = license_analysis.collect_hf_license_candidates(
        FakeClient(),
        "https://huggingface.co/acme/model",
    )
    assert "MIT" in meta
    assert "Apache-2.0" in meta
    assert "GPL-3.0-only" in readme


def test_load_json_list_validates_shape(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a list"):
        license_analysis._load_json_list(path)
