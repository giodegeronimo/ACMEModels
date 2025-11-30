"""License compatibility metric for Hugging Face models."""

from __future__ import annotations

import json
import logging
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from src.clients.hf_client import HFClient
from src.metrics.base import Metric, MetricOutput
from src.utils.env import fail_stub_active

_LOGGER = logging.getLogger(__name__)

FAIL = False

_DEFAULT_URL = "https://huggingface.co/google-bert/bert-base-uncased"

_FAILURE_VALUES: Dict[str, float] = {
    "https://huggingface.co/google-bert/bert-base-uncased": 0.1,
    "https://huggingface.co/parvk11/audience_classifier_model": 0.8,
    "https://huggingface.co/openai/whisper-tiny/tree/main": 0.9,
}

COMPAT_WEIGHT = 0.8
CLARITY_WEIGHT = 0.2

_LICENSE_PATTERNS: Dict[str, re.Pattern[str]] = {
    "MIT": re.compile(r"\bmit\b", re.IGNORECASE),
    "Apache-2.0": re.compile(r"apache\s*-?\s*2(?:\.0)?", re.IGNORECASE),
    "BSD-2-Clause": re.compile(r"bsd\s*2\s*-?clause", re.IGNORECASE),
    "BSD-3-Clause": re.compile(r"bsd\s*3\s*-?clause", re.IGNORECASE),
    "MPL-2.0": re.compile(r"mpl\s*-?\s*2(?:\.0)?|mozilla", re.IGNORECASE),
    "LGPL-2.1-only": re.compile(r"lgpl\s*(v?2\.1|2\.1)\b", re.IGNORECASE),
    "LGPL-3.0-only": re.compile(r"lgpl\s*(v?3|3\.0)\b", re.IGNORECASE),
    "CC-BY-4.0": re.compile(r"cc-?by-?4(?:\.0)?", re.IGNORECASE),
    "CC0-1.0": re.compile(r"cc0|public\s+domain", re.IGNORECASE),
    "CC-BY-SA-4.0": re.compile(r"cc-?by-?sa-?4(?:\.0)?", re.IGNORECASE),
    "GPL-3.0-only": re.compile(r"gpl\s*(v?3|3\.0)\b", re.IGNORECASE),
    "AGPL-3.0-only": re.compile(r"agpl\s*(v?3|3\.0)\b", re.IGNORECASE),
    "OpenRAIL-M": re.compile(r"openrail[-_ ]?m", re.IGNORECASE),
    "Custom": re.compile(r"custom\s+license", re.IGNORECASE),
}


class _HFClientProtocol(Protocol):
    def get_model_info(self, repo_id: str) -> Any: ...

    def get_model_readme(self, repo_id: str) -> str: ...


class LicenseMetric(Metric):
    """License clarity and compatibility score.

    Uses policy files under ``data/`` to categorize SPDX-like license
    identifiers. Prefers HF metadata; falls back to README parsing.
    """

    def __init__(self, hf_client: Optional[_HFClientProtocol] = None) -> None:
        super().__init__(name="License Score", key="license")
        self._hf: _HFClientProtocol = hf_client or HFClient()

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        hf_url = _extract_hf_url(url_record)
        if fail_stub_active(FAIL):
            time.sleep(0.05)
            url = hf_url or _DEFAULT_URL
            return _FAILURE_VALUES.get(url, _FAILURE_VALUES[_DEFAULT_URL])

        if not hf_url:
            return 0.0

        policy = _load_policy()
        from_meta, from_readme = _collect_candidates(self._hf, hf_url)
        candidates = _normalize_candidates([*from_meta, *from_readme])

        _LOGGER.info(
            "License metric inputs for %s: metadata=%s readme=%s "
            "normalized=%s",
            hf_url,
            from_meta,
            from_readme,
            candidates,
        )

        if not candidates:
            _LOGGER.info(
                "License metric: no recognized licenses for %s",
                hf_url,
            )
            return 0.0

        classification = _classify(candidates, policy)
        compat = 0.0
        if classification == "compatible":
            compat = 1.0
        elif classification == "caution":
            compat = 0.5
        elif classification == "incompatible":
            compat = 0.0
        else:
            compat = 0.0

        final = max(0.0, min(compat, 1.0))
        _LOGGER.info(
            "License metric for %s: class=%s score=%.2f",
            hf_url,
            classification,
            final,
        )
        return final


def _extract_hf_url(record: Dict[str, str]) -> Optional[str]:
    return record.get("hf_url")


def _collect_candidates(
    hf: _HFClientProtocol, hf_url: str
) -> Tuple[List[str], List[str]]:
    meta: List[str] = []
    readme: List[str] = []

    info = _safe_model_info(hf, hf_url)
    if info is not None:
        top = getattr(info, "license", None)
        meta.extend(_split_license_field(top))
        card = getattr(info, "card_data", None)
        if isinstance(card, dict):
            meta.extend(_split_license_field(card.get("license")))

    text = _safe_readme(hf, hf_url)
    if text:
        section = _extract_license_section(text)
        candidates = _find_licenses_in_text(section or text)
        readme.extend(candidates)

    return meta, readme


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
    policy = _load_policy()
    for slug in policy.all_slugs:
        if re.search(rf"\b{re.escape(slug)}\b", text, re.IGNORECASE):
            found.append(slug)
    return list(dict.fromkeys(found))


def _normalize_candidates(candidates: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for item in candidates:
        slug = _normalize_slug(item)
        if slug:
            normalized.append(slug)
    return list(dict.fromkeys(normalized))


def _normalize_slug(value: str) -> Optional[str]:
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
        'bsd 3-clause "new" or "revised" license': "BSD-3-Clause",
        "bsd-3-clause": "BSD-3-Clause",
        "bsd 2-clause": "BSD-2-Clause",
        'bsd 2-clause "simplified" license': "BSD-2-Clause",
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
        "cc-by-sa-4.0": "CC-BY-SA-4.0",
        "creative commons attribution-noncommercial 4.0 international":
            "CC-BY-NC-4.0",
        "cc-by-nc-4.0": "CC-BY-NC-4.0",
        "creative commons attribution-noderivatives 4.0 international":
            "CC-BY-ND-4.0",
        "cc-by-nd-4.0": "CC-BY-ND-4.0",
        "openrail-m": "OpenRAIL-M",
        "openrail++": "OpenRAIL++",
        "custom": "Custom",
    }
    if s in synonyms:
        return synonyms[s]
    spdx_like = re.sub(r"\s+", "-", value.strip())
    return spdx_like


class _LicensePolicy:
    def __init__(
        self,
        compatible: Sequence[str],
        caution: Sequence[str],
        incompatible: Sequence[str],
    ) -> None:
        self.compatible = {x.lower(): x for x in compatible}
        self.caution = {x.lower(): x for x in caution}
        self.incompatible = {x.lower(): x for x in incompatible}
        self.all_slugs = set(
            [
                *self.compatible.values(),
                *self.caution.values(),
                *self.incompatible.values(),
            ]
        )

    def class_of(self, slug: str) -> str:
        key = slug.lower()
        if key in self.compatible:
            return "compatible"
        if key in self.incompatible:
            return "incompatible"
        if key in self.caution:
            return "caution"
        return "unknown"


@lru_cache(maxsize=1)
def _load_policy() -> _LicensePolicy:
    base = Path(__file__).resolve().parents[2].with_name("data")
    try:
        compatible = _load_json_list(base / "licenses_compatible_spdx.json")
        caution = _load_json_list(base / "licenses_caution_spdx.json")
        incompatible = _load_json_list(
            base / "licenses_incompatible_spdx.json"
        )
    except Exception as exc:  # pragma: no cover - defensive
        _LOGGER.debug("Falling back to defaults: %s", exc)
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
    return _LicensePolicy(compatible, caution, incompatible)


def _load_json_list(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Policy file {path} must contain a list.")
    return [str(x) for x in data]


def _classify(candidates: Sequence[str], policy: _LicensePolicy) -> str:
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


def _safe_model_info(hf: _HFClientProtocol, hf_url: str) -> Optional[Any]:
    try:
        return hf.get_model_info(hf_url)
    except Exception as exc:
        _LOGGER.debug("Model info unavailable for %s: %s", hf_url, exc)
        return None


def _safe_readme(hf: _HFClientProtocol, hf_url: str) -> str:
    try:
        return hf.get_model_readme(hf_url)
    except Exception as exc:
        _LOGGER.debug("README unavailable for %s: %s", hf_url, exc)
        return ""
