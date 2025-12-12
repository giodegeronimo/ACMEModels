"""Helpers for license analysis (extracted for LicenseMetric)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_LICENSE_PATTERNS: Dict[str, re.Pattern[str]] = {
    "MIT": re.compile(r"\bMIT\b", re.IGNORECASE),
    "Apache-2.0": re.compile(r"apache\s*(license)?\s*2\.0", re.IGNORECASE),
    "GPL-3.0-only": re.compile(r"\bGPL[-\s]?3(?:\.0)?\b", re.IGNORECASE),
    "CC-BY-SA-4.0": re.compile(r"CC[-\s]?BY[-\s]?SA[-\s]?4\.0", re.IGNORECASE),
    "CC-BY-NC-4.0": re.compile(r"CC[-\s]?BY[-\s]?NC[-\s]?4\.0", re.IGNORECASE),
    "CC-BY-ND-4.0": re.compile(r"CC[-\s]?BY[-\s]?ND[-\s]?4\.0", re.IGNORECASE),
}


@dataclass
class _LicensePolicy:
    compatible: Dict[str, str]
    caution: Dict[str, str]
    incompatible: Dict[str, str]
    all_slugs: set[str]

    def class_of(self, slug: str) -> str:
        key = slug.lower()
        if key in self.compatible:
            return "compatible"
        if key in self.incompatible:
            return "incompatible"
        if key in self.caution:
            return "caution"
        return "unknown"


def collect_hf_license_candidates(
    hf_client: Any, hf_url: str
) -> Tuple[List[str], List[str]]:
    """Collect license candidates from HF model info and README."""

    meta: List[str] = []
    readme: List[str] = []

    info = _safe_model_info(hf_client, hf_url)
    if info is not None:
        top = getattr(info, "license", None)
        meta.extend(_split_license_field(top))
        card = getattr(info, "card_data", None)
        if isinstance(card, dict):
            meta.extend(_split_license_field(card.get("license")))

    text = _safe_readme(hf_client, hf_url)
    if text:
        section = _extract_license_section(text)
        candidates = _find_licenses_in_text(section or text)
        readme.extend(candidates)

    return meta, readme


def normalize_license_candidates(candidates: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for item in candidates:
        slug = _normalize_slug(item)
        if slug:
            normalized.append(slug)
    return list(dict.fromkeys(normalized))


def evaluate_classification(
    candidates: Sequence[str], policy: _LicensePolicy
) -> str:
    if not candidates:
        return "unknown"
    classes = {policy.class_of(slug) for slug in candidates}
    classes.discard("unknown")
    if not classes:
        return "unknown"
    if classes == {"compatible"}:
        return "compatible"
    if "incompatible" in classes and "compatible" not in classes:
        return "incompatible"
    if "compatible" in classes:
        return "caution"
    if "caution" in classes:
        return "caution"
    return "unknown"


def load_license_policy() -> _LicensePolicy:
    base = Path(__file__).resolve().parents[1].with_name("data")
    try:
        compatible = _load_json_list(base / "licenses_compatible_spdx.json")
        caution = _load_json_list(base / "licenses_caution_spdx.json")
        incompatible = _load_json_list(
            base / "licenses_incompatible_spdx.json"
        )
    except Exception:
        compatible = [
            "MIT",
            "Apache-2.0",
            "BSD-2-Clause",
            "BSD-3-Clause",
            "MPL-2.0",
            "LGPL-2.1-only",
            "LGPL-3.0-only",
            "CC-BY-4.0",
            "CC0-1.0",
        ]
        caution = ["CC-BY-SA-4.0", "OpenRAIL-M", "Custom"]
        incompatible = [
            "GPL-3.0-only",
            "AGPL-3.0-only",
            "CC-BY-NC-4.0",
            "CC-BY-ND-4.0",
        ]
    return _LicensePolicy(
        compatible={x.lower(): x for x in compatible},
        caution={x.lower(): x for x in caution},
        incompatible={x.lower(): x for x in incompatible},
        all_slugs=set([*compatible, *caution, *incompatible]),
    )


def _split_license_field(field: Any) -> List[str]:
    results: List[str] = []
    if not field:
        return results
    if isinstance(field, str):
        parts = re.split(r"\s+(?:OR|AND|/|,|\|\|)\s+", field, flags=re.I)
        results.extend([part.strip() for part in parts if part.strip()])
    elif isinstance(field, list):
        for item in field:
            results.extend(_split_license_field(item))
    return results


def _extract_license_section(text: str) -> str:
    heading = re.search(
        r"^#{1,6}\s+license\b.*$", text, re.IGNORECASE | re.MULTILINE
    )
    if not heading:
        return ""
    start = heading.end()
    next_heading = re.search(
        r"^#{1,6}\s+\S+.*$", text[start:], re.MULTILINE
    )
    end = start + (next_heading.start() if next_heading else len(text))
    return text[start:end]


def _find_licenses_in_text(text: str) -> List[str]:
    if not text:
        return []
    found: List[str] = []
    for slug, pattern in _LICENSE_PATTERNS.items():
        if pattern.search(text):
            found.append(slug)
    for slug in _LICENSE_PATTERNS.keys():
        if re.search(rf"\b{re.escape(slug)}\b", text, re.IGNORECASE):
            found.append(slug)
    return list(dict.fromkeys(found))


def _normalize_slug(value: str) -> str | None:
    s = value.strip().lower()
    if not s:
        return None
    synonyms: Dict[str, str] = {
        "mit license": "MIT",
        "mit": "MIT",
        "apache license 2.0": "Apache-2.0",
        "apache license, version 2.0": "Apache-2.0",
        "apache2": "Apache-2.0",
        "apache 2.0": "Apache-2.0",
        "apache-2.0": "Apache-2.0",
        "bsd 3-clause": "BSD-3-Clause",
        "bsd-3-clause": "BSD-3-Clause",
        "bsd 2-clause": "BSD-2-Clause",
        "bsd-2-clause": "BSD-2-Clause",
        "mozilla public license 2.0": "MPL-2.0",
        "mpl 2.0": "MPL-2.0",
        "mpl-2.0": "MPL-2.0",
        "gnu lesser general public license v2.1": "LGPL-2.1-only",
        "lgpl 2.1": "LGPL-2.1-only",
        "lgpl v2.1": "LGPL-2.1-only",
        "lgpl-2.1": "LGPL-2.1-only",
        "gnu lesser general public license v3.0": "LGPL-3.0-only",
        "lgpl3": "LGPL-3.0-only",
        "lgpl v3": "LGPL-3.0-only",
        "lgpl-3.0": "LGPL-3.0-only",
        "gnu general public license v3.0": "GPL-3.0-only",
        "gpl v3": "GPL-3.0-only",
        "gpl-3.0": "GPL-3.0-only",
        "gnu affero general public license v3.0": "AGPL-3.0-only",
        "agpl-3.0": "AGPL-3.0-only",
        "cc-by 4.0": "CC-BY-4.0",
        "creative commons attribution 4.0 international": "CC-BY-4.0",
        "cc-by-4.0": "CC-BY-4.0",
        "creative commons zero v1.0 universal": "CC0-1.0",
        "cc0": "CC0-1.0",
        "cc0-1.0": "CC0-1.0",
        "cc-by-sa 4.0": "CC-BY-SA-4.0",
        "creative commons attribution share alike 4.0 international":
            "CC-BY-SA-4.0",
        "cc-by-nc 4.0": "CC-BY-NC-4.0",
        "cc-by-nd 4.0": "CC-BY-ND-4.0",
        "openrail-m": "OpenRAIL-M",
        "openrail++": "OpenRAIL++",
        "custom": "Custom",
    }
    if s in synonyms:
        return synonyms[s]
    spdx_like = re.sub(r"\s+", "-", value.strip())
    return spdx_like


def _safe_model_info(hf_client: Any, hf_url: str) -> Any | None:
    try:
        return hf_client.get_model_info(hf_url)
    except Exception:
        return None


def _safe_readme(hf_client: Any, hf_url: str) -> str:
    try:
        return hf_client.get_model_readme(hf_url)
    except Exception:
        return ""


def _load_json_list(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Policy file {path} must contain a list.")
    return [str(x) for x in data]
