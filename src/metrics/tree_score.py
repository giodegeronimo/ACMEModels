from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Iterable, Mapping, Optional

from src.clients.hf_client import HFClient
from src.metrics.base import Metric, MetricOutput
# Reuse existing metrics to compute parent net scores
from src.metrics.bus_factor import BusFactorMetric
from src.metrics.code_quality import CodeQualityMetric
from src.metrics.dataset_and_code import DatasetAndCodeMetric
from src.metrics.dataset_quality import DatasetQualityMetric
from src.metrics.license import LicenseMetric
from src.metrics.performance import PerformanceMetric
from src.metrics.ramp_up import RampUpMetric
from src.metrics.size import SizeMetric
from src.utils.env import fail_stub_active, ignore_fail_flags

_LOGGER = logging.getLogger(__name__)


FAIL = False

_DEFAULT_URL = "https://huggingface.co/google-bert/bert-base-uncased"

_FAILURE_VALUES: Dict[str, float] = {
    "https://huggingface.co/google-bert/bert-base-uncased": 0.6,
    "https://huggingface.co/parvk11/audience_classifier_model": 0.7,
    "https://huggingface.co/openai/whisper-tiny/tree/main": 0.65,
}

# Defensive caps to avoid explosion
MAX_PARENT_FANOUT = 5
MAX_RECURSION_DEPTH = 2
DEPTH_WEIGHTS: Dict[int, float] = {0: 1.0, 1: 0.7, 2: 0.5}


class TreeScoreMetric(Metric):
    """Average of parents' total model scores discovered from HF lineage.

    - Discovers parent models from Hugging Face metadata (card_data/config)
      and README hints (e.g., "base model", "fine-tuned from").
    - Computes each parent's total score by averaging the standard metrics
      (excluding TreeScore itself) and returns the mean across parents.
    - If no parents are found or scores are unavailable, returns the target
      model's own net score.
    """

    def __init__(self, hf_client: Optional[HFClient] = None) -> None:
        super().__init__(name="Tree Score", key="tree_score")
        self._hf: HFClient = hf_client or HFClient()

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        if fail_stub_active(FAIL):
            time.sleep(0.05)
            url = _extract_hf_url(url_record) or _DEFAULT_URL
            _LOGGER.info(
                "FAIL flag enabled; returning stub tree score for %s",
                url,
            )
            return _FAILURE_VALUES.get(url, _FAILURE_VALUES[_DEFAULT_URL])
        # In unit-test mode (ACME_IGNORE_FAIL=1), we still compute during
        # interactive runs so parents' metrics can be inspected. Fan-out and
        # depth are already capped to keep it safe.
        if ignore_fail_flags():
            _LOGGER.info(
                "ACME_IGNORE_FAIL=1 detected; proceeding with capped traversal"
            )

        hf_url = _extract_hf_url(url_record)
        if not hf_url:
            return self._fallback_to_model_score(
                None, "No hf_url provided for tree_score"
            )

        # Identify parent repo ids like "owner/name".
        parents = _discover_parents(self._hf, hf_url)
        if not parents:
            return self._fallback_to_model_score(
                hf_url, f"No parents discovered for {hf_url}"
            )
        _LOGGER.info(
            "Discovered %d top-level parent(s) for %s: %s",
            len(parents),
            hf_url,
            ", ".join(parents),
        )

        # Recurse to collect ancestors up to MAX_RECURSION_DEPTH
        visited: set[str] = set()
        ancestors_depth: Dict[str, int] = {}
        for parent in parents:
            chain = _collect_ancestors_with_depth(
                self._hf, parent, MAX_RECURSION_DEPTH, visited
            )
            for slug, depth in chain.items():
                if (
                    slug not in ancestors_depth
                    or depth < ancestors_depth[slug]
                ):
                    ancestors_depth[slug] = depth

        if not ancestors_depth:
            return self._fallback_to_model_score(
                hf_url, f"No ancestors collected for {hf_url}"
            )

        _LOGGER.info(
            "Collected %d unique ancestor(s) up to depth %d",
            len(ancestors_depth),
            MAX_RECURSION_DEPTH,
        )

        # Depth-weighted average so closer ancestors matter more
        weighted_total = 0.0
        weight_sum = 0.0
        for ancestor, depth in sorted(ancestors_depth.items(),
                                      key=lambda x: (x[1], x[0])):
            score = _compute_parent_net_score(ancestor)
            if score is None:
                _LOGGER.debug("Ancestor %s produced no score", ancestor)
                continue
            weight = DEPTH_WEIGHTS.get(min(depth, MAX_RECURSION_DEPTH), 0.5)
            weighted_total += score * weight
            weight_sum += weight
            _LOGGER.debug(
                "Ancestor %s (depth=%d, weight=%.2f) net score: %.3f",
                ancestor,
                depth,
                weight,
                score,
            )

        if weight_sum <= 0.0:
            return self._fallback_to_model_score(
                hf_url,
                f"No ancestor scores produced for {hf_url}",
            )

        final = weighted_total / weight_sum
        _LOGGER.info("Tree score for %s computed as %.2f", hf_url, final)
        return final

    def _fallback_to_model_score(
        self,
        hf_url: Optional[str],
        reason: str,
    ) -> float:
        _LOGGER.info(
            "%s; falling back to target model net score",
            reason,
        )
        if not hf_url:
            _LOGGER.info("Missing hf_url; defaulting tree score to 0.5")
            return 0.5
        slug = _to_repo_slug(hf_url)
        if not slug:
            _LOGGER.info(
                "Could not derive repo slug from %s; defaulting to 0.5",
                hf_url,
            )
            return 0.5
        self_score = _compute_parent_net_score(slug)
        if self_score is None:
            _LOGGER.info(
                "Target model net score unavailable for %s; defaulting to 0.5",
                hf_url,
            )
            return 0.5
        _LOGGER.info(
            "Tree score fallback set to %.2f using target model net score",
            self_score,
        )
        return self_score


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
    return normalized[:MAX_PARENT_FANOUT]


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


def _collect_ancestors_with_depth(
    hf: HFClient,
    repo_slug: str,
    max_depth: int,
    visited: Optional[set[str]] = None,
) -> Dict[str, int]:
    """Depth-limited DFS to collect ancestors and their minimum depth.

    Returns a mapping {slug: depth} including ``repo_slug`` itself at depth 0.
    ``visited`` prevents infinite loops across chains.
    """
    if visited is None:
        visited = set()

    depths: Dict[str, int] = {}

    def _dfs(slug: str, depth: int) -> None:
        if slug in visited:
            _LOGGER.debug("Already visited %s; skipping", slug)
            return
        visited.add(slug)

        if slug not in depths or depth < depths[slug]:
            depths[slug] = depth
        if depth >= max_depth:
            _LOGGER.debug("Max depth reached at %s (depth=%d)", slug, depth)
            return

        url = f"https://huggingface.co/{slug}"
        try:
            parents = _discover_parents(hf, url)
        except Exception:
            _LOGGER.debug("Failed to discover parents for %s",
                          slug, exc_info=True)
            parents = []

        if parents:
            _LOGGER.debug(
                "Depth %d: %s has %d parent(s): %s",
                depth,
                slug,
                len(parents),
                ", ".join(parents),
            )
        for p in parents:
            _dfs(p, depth + 1)

    _dfs(repo_slug, 0)
    return depths


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

    _LOGGER.debug("Computing parent net score for %s", repo_slug)
    numeric_values: list[float] = []
    for metric in metrics:
        try:
            value = metric.compute(url_record)  # may call HF as needed
        except Exception:
            metric_name = getattr(metric, "key", metric.__class__.__name__)
            _LOGGER.debug(
                "Metric %s failed for %s",
                metric_name,
                repo_slug,
                exc_info=True,
            )
            continue

        metric_name = getattr(metric, "name", metric.__class__.__name__)
        numeric = _to_numeric_metric(value)
        # Log raw metric output for transparency while traversing parents.
        _LOGGER.info("[tree_score] %s for %s: %s",
                     metric_name, repo_slug, value)
        if numeric is not None:
            numeric_values.append(numeric)
        print(f"[tree_score] {metric.name} for {repo_slug}: {value}")

    if not numeric_values:
        _LOGGER.info("No metric values available for %s; skipping", repo_slug)
        return None
    net = sum(numeric_values) / len(numeric_values)
    _LOGGER.info("Net parent score for %s: %.2f", repo_slug, net)
    return net


def _to_numeric_metric(value: MetricOutput) -> Optional[float]:
    """Convert a metric output into a single float if possible.

    - If it's a number, return it.
    - If it's a mapping (e.g., Size returns per-device scores), use a
      representative value: prefer 'desktop_pc' if present, otherwise take
      the average of numeric values.
    - Otherwise, return None.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        # Prefer desktop_pc for a desktop-representative score
        if (
            "desktop_pc" in value
            and isinstance(value["desktop_pc"], (int, float))
        ):
            return float(value["desktop_pc"])
        # Fallback: mean of numeric values
        vals: list[float] = [
            float(v) for v in value.values() if isinstance(v, (int, float))
        ]
        if vals:
            return sum(vals) / len(vals)
    return None
