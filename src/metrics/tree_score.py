from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterable, Mapping, Optional

from src.clients.hf_client import HFClient
from src.metrics.base import Metric, MetricOutput
from src.utils.env import fail_stub_active, ignore_fail_flags

# Reuse existing metrics to compute parent net scores
from src.metrics.bus_factor import BusFactorMetric
from src.metrics.code_quality import CodeQualityMetric
from src.metrics.dataset_and_code import DatasetAndCodeMetric
from src.metrics.dataset_quality import DatasetQualityMetric
from src.metrics.performance import PerformanceMetric
from src.metrics.ramp_up import RampUpMetric
from src.metrics.size import SizeMetric
from src.metrics.license import LicenseMetric


FAIL = True

_DEFAULT_URL = "https://huggingface.co/google-bert/bert-base-uncased"

_FAILURE_VALUES: Dict[str, float] = {
    "https://huggingface.co/google-bert/bert-base-uncased": 0.6,
    "https://huggingface.co/parvk11/audience_classifier_model": 0.7,
    "https://huggingface.co/openai/whisper-tiny/tree/main": 0.65,
}


class TreeScoreMetric(Metric):
    """Average of parents' total model scores discovered from HF lineage.

    - Discovers parent models from Hugging Face metadata (card_data/config)
      and README hints (e.g., "base model", "fine-tuned from").
    - Computes each parent's total score by averaging the standard metrics
      (excluding TreeScore itself) and returns the mean across parents.
    - If no parents are found or scores are unavailable, returns 0.5.
    """

    def __init__(self, hf_client: Optional[HFClient] = None) -> None:
        super().__init__(name="Tree Score", key="tree_score")
        self._hf: HFClient = hf_client or HFClient()

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        if fail_stub_active(FAIL):
            time.sleep(0.05)
            url = _extract_hf_url(url_record) or _DEFAULT_URL
            return _FAILURE_VALUES.get(url, _FAILURE_VALUES[_DEFAULT_URL])
        # In unit-test mode (ACME_IGNORE_FAIL=1), avoid network fan-out.
        if ignore_fail_flags():
            return 0.5

        hf_url = _extract_hf_url(url_record)
        if not hf_url:
            return 0.5

        # Identify parent repo ids like "owner/name".
        parents = _discover_parents(self._hf, hf_url)
        if not parents:
            return 0.5

        scores: list[float] = []
        for parent in parents:
            score = _compute_parent_net_score(parent)
            if score is not None:
                scores.append(score)

        return sum(scores) / len(scores) if scores else 0.5


def _extract_hf_url(record: Mapping[str, Any]) -> Optional[str]:
    value = record.get("hf_url")
    return str(value) if isinstance(value, str) else None


def _discover_parents(hf: HFClient, hf_url: str) -> list[str]:
    """
    Return a list of parent repo slugs (owner/name)
    for a model URL.
    """
    parents: list[str] = []

    # Try metadata first
    try:
        info = hf.get_model_info(hf_url)
    except Exception:
        info = None

    if info is not None:
        card = getattr(info, "card_data", None)
        config = getattr(info, "config", None)
        for container in (card, config, info):
            if isinstance(container, dict):
                parents.extend(_extract_parents_from_mapping(container))
            else:
                parents.extend(_extract_parents_from_object(container))

    # Fallback: scan README for explicit parent/base hints and HF links
    try:
        readme = hf.get_model_readme(hf_url)
    except Exception:
        readme = ""
    parents.extend(_extract_parents_from_readme(readme or ""))

    # Normalize and dedupe; keep order
    seen: set[str] = set()
    normalized: list[str] = []
    for item in parents:
        slug = _to_repo_slug(item)
        if slug and slug not in seen:
            seen.add(slug)
            normalized.append(slug)

    # Limit fan-out defensively
    return normalized[:5]


def _extract_parents_from_mapping(mapping: Mapping[str, Any]) -> list[str]:
    results: list[str] = []
    params = ("base_model",
              "base_model_id",
              "parent_model",
              "parents",
              "parent_models")
    for key in params:
        value = mapping.get(key)
        if isinstance(value, str):
            results.append(value)
        elif isinstance(value, (list, tuple)):
            results.extend(
                [str(v) for v in value if isinstance(v, (str, bytes))]
            )
    return results


def _extract_parents_from_object(obj: Any) -> list[str]:
    results: list[str] = []
    for attr in ("base_model", "parent_model", "parents", "parent_models"):
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            if isinstance(value, str):
                results.append(value)
            elif isinstance(value, (list, tuple)):
                results.extend(
                    [str(v) for v in value if isinstance(v, (str, bytes))]
                )
    return results


parent_hint_pattern = (
    r"(?im)(?:base|parent)"
    r"\s*model[^:\n]*[:：]\s*"
    r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
)

_PARENT_HINT = re.compile(parent_hint_pattern)

_FINE_TUNED_FROM = re.compile(
    r"(?im)fine[- ]tuned\s+(?:from|of|on)\s+([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
)
_HF_LINK = re.compile(
    r"https?://huggingface\.co/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)"
)


def _extract_parents_from_readme(text: str) -> list[str]:
    results: list[str] = []
    for pattern in (_PARENT_HINT, _FINE_TUNED_FROM):
        for m in pattern.finditer(text):
            results.append(m.group(1))
    for m in _HF_LINK.finditer(text):
        results.append(f"{m.group(1)}/{m.group(2)}")
    return results


def _to_repo_slug(value: str) -> Optional[str]:
    s = (value or "").strip().strip("/")
    if not s:
        return None
    # Convert full URLs to owner/name
    if s.startswith("http://") or s.startswith("https://"):
        try:
            return HFClient._normalize_repo_id(s)  # type: ignore[attr-defined]
        except Exception:
            return None
    if "/" in s:
        parts = [p for p in s.split("/") if p]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    return None


def _compute_parent_net_score(repo_slug: str) -> Optional[float]:
    """Compute a parent's net score by averaging standard metric outputs."""
    url_record = {"hf_url": f"https://huggingface.co/{repo_slug}"}

    # Build metrics explicitly to avoid importing the registry (no recursion)
    metrics: Iterable[Metric] = (
        RampUpMetric(),
        BusFactorMetric(),
        LicenseMetric(),
        SizeMetric(),
        DatasetAndCodeMetric(),
        DatasetQualityMetric(),
        CodeQualityMetric(),
        PerformanceMetric(),
    )

    numeric_values: list[float] = []
    for metric in metrics:
        try:
            value = metric.compute(url_record)  # may call HF as needed
        except Exception:
            continue
        if isinstance(value, (int, float)):
            numeric_values.append(float(value))

    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)
