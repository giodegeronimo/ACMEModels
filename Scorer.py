# Scorer.py
"""
Scorer: compute all required metrics + latencies for one resource (usually a MODEL),
and return a single dict ready for OutputFormatter.write_line().

Design goals
------------
- Parallel metric computation (ThreadPoolExecutor), bounded by available cores.
- Pluggable metric registry (add new metrics without touching the orchestrator).
- No stdout logging here (keep machine output clean). Optional file logging via LOG_* envs.
- Defensive: never raise; on error, return score=0.0 and latency>=0 for that metric.

Inputs
------
- A Resource (from URL_Fetcher.determineResource) + its metadata + README (best-effort).
- We primarily score MODELs per the Phase-1 I/O, but code supports other categories too.

Outputs (Table 1 fields)
------------------------
name, category, net_score, net_score_latency,
<metric>, <metric>_latency for:
  - ramp_up_time
  - bus_factor
  - performance_claims
  - license
  - size_score (object) + size_score_latency
  - dataset_and_code_score
  - dataset_quality
  - code_quality
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Tuple

# Import your teammate modules
# NOTE: file name is "URL_Fetcher.py" (with 'a'); import accordingly.
from URL_Fetcher import (  # type: ignore
    Resource,
    ModelResource,
    DatasetResource,
    CodeResource,
    UrlCategory,
    hasLicenseSection,
)

# ------------------------- Logging (silent by default) ------------------------- #

import logging


def _make_logger() -> logging.Logger:
    logger = logging.getLogger("scorer")
    if getattr(logger, "_configured", False):
        return logger

    level_env = os.environ.get("LOG_LEVEL", "0").strip()
    try:
        level_num = int(level_env)
    except ValueError:
        level_num = 0

    if level_num <= 0:
        level = logging.CRITICAL + 1  # effectively silent
    elif level_num == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logger.setLevel(level)
    logger.propagate = False

    handler: logging.Handler
    log_file = os.environ.get("LOG_FILE")
    handler = logging.FileHandler(log_file) if log_file else logging.NullHandler()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | scorer | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    setattr(logger, "_configured", True)
    return logger


LOG = _make_logger()

# ------------------------- Utilities ------------------------- #


def _now_ms() -> int:
    return int(time.perf_counter() * 1000)


def _latency_wrapper(fn: Callable[[], float | Dict[str, float]]) -> Tuple[float | Dict[str, float], int]:
    """
    Run fn(), measure latency (ms). On error, return default score 0.0 (or {} for size_score).
    """
    t0 = _now_ms()
    try:
        val = fn()
    except Exception as e:  # never crash a metric
        LOG.debug(f"metric error: {e}")
        # Heuristic: if fn is for size_score (dict), return {}; else 0.0
        val = {} if fn.__name__.endswith("size_score") else 0.0  # type: ignore[assignment]
    lat = max(0, _now_ms() - t0)
    return val, lat


def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _bool_to_score(ok: bool) -> float:
    return 1.0 if ok else 0.0


# ------------------------- Metric implementations (starter heuristics) ------------------------- #
# These are intentionally lightweight so Phase-1 runs fast and is easy to extend later.
# You can improve each metric by analyzing repo files, Git metadata, etc., without changing the interface.


@dataclass(frozen=True)
class Inputs:
    """What each metric may need."""
    resource: Resource
    metadata: Dict[str, Any]
    readme: str | None


# Ramp-up time: docs/examples/freshness signals from HF metadata + README presence.
def metric_ramp_up_time(inp: Inputs) -> float:
    md = inp.metadata or {}
    readme_ok = bool(inp.readme and len(inp.readme) > 200)  # crude, but stable
    likes = md.get("likes") or 0
    file_count = md.get("fileCount") or 0
    freshness_bonus = 0.0
    # HF 'lastModified' ISO string -> recent => bonus
    lm = md.get("lastModified")
    if isinstance(lm, str) and re.search(r"20\d{2}-\d{2}-\d{2}", lm):
        freshness_bonus = 0.15

    # Weighted blend (keep in [0,1])
    base = 0.4 * _bool_to_score(readme_ok) + 0.3 * min(1.0, file_count / 10.0) + 0.3 * min(1.0, likes / 50.0)
    return _clamp01(base + freshness_bonus)


# Bus factor: proxy using file_count (more artifacts) and recent activity.
def metric_bus_factor(inp: Inputs) -> float:
    md = inp.metadata or {}
    file_count = md.get("fileCount") or 0
    freshness = 0.0
    lm = md.get("lastModified")
    if isinstance(lm, str) and re.search(r"20\d{2}-\d{2}-\d{2}", lm):
        freshness = 1.0
    # Simple proxy (Phase-1): more artifacts + recent update => safer
    score = 0.6 * min(1.0, file_count / 8.0) + 0.4 * freshness
    return _clamp01(score)


# Performance claims: look for "Benchmark", "results", tables, or metrics in the README.
_BENCH_RE = re.compile(
    r"(benchmark|results?|accuracy|f1|bleu|rouge|mmlu|leaderboard|eval|evaluation)",
    re.IGNORECASE,
)


def metric_performance_claims(inp: Inputs) -> float:
    text = inp.readme or ""
    if not text:
        return 0.0
    # Very light heuristic: keywords + presence of a Markdown table
    has_keywords = bool(_BENCH_RE.search(text))
    has_table = "|" in text and "---" in text
    return _clamp01(0.8 * _bool_to_score(has_keywords) + 0.2 * _bool_to_score(has_table))


# License: explicit "License" section in README for MODEL/DATASET; for GitHub code, SPDX proxy if present.
def metric_license(inp: Inputs) -> float:
    # For Phase-1, just check README heading exists.
    has_section = hasLicenseSection(inp.readme)
    return 1.0 if has_section else 0.0


# Size score (object): map to hardware buckets with a conservative heuristic.
# Without downloading weights in Phase-1, use file_count as a cheap proxy.
def metric_size_score(inp: Inputs) -> Dict[str, float]:
    md = inp.metadata or {}
    file_count = md.get("fileCount") or 0
    # Suppose fewer artifacts => smaller model => broader deployability (very rough starter).
    # Tweak later to inspect config/architecture or actual weight sizes.
    smallish = 1.0 if file_count <= 5 else (0.6 if file_count <= 15 else 0.2)
    medium = 0.8 if file_count <= 10 else (0.5 if file_count <= 25 else 0.3)
    large = 0.6 if file_count <= 15 else (0.4 if file_count <= 40 else 0.2)
    server = 0.9 if file_count <= 40 else 0.7

    return {
        "raspberry_pi": _clamp01(smallish),
        "jetson_nano": _clamp01(medium),
        "desktop_pc": _clamp01(large),
        "aws_server": _clamp01(server),
    }


# Dataset & code availability: look for dataset/code links in README.
_DATASET_LINK_RE = re.compile(r"https?://huggingface\.co/(datasets/|.*\bdata)", re.IGNORECASE)
_CODE_LINK_RE = re.compile(r"https?://(github\.com|gitlab\.com)/", re.IGNORECASE)


def metric_dataset_and_code_score(inp: Inputs) -> float:
    text = inp.readme or ""
    has_dataset = bool(_DATASET_LINK_RE.search(text))
    has_code = bool(_CODE_LINK_RE.search(text))
    return _clamp01(0.5 * _bool_to_score(has_dataset) + 0.5 * _bool_to_score(has_code))


# Dataset quality: look for a "Dataset" heading + minimal bullets/fields.
_DATASET_HDR_RE = re.compile(r"^\s*#{1,6}\s*dataset(s)?\b", re.IGNORECASE | re.MULTILINE)


def metric_dataset_quality(inp: Inputs) -> float:
    text = inp.readme or ""
    if not text:
        return 0.0
    has_hdr = bool(_DATASET_HDR_RE.search(text))
    has_bullets = text.count("\n- ") + text.count("\n* ") >= 3
    freshness_bonus = 0.1 if (inp.metadata or {}).get("downloads", 0) > 1000 else 0.0
    return _clamp01(0.6 * _bool_to_score(has_hdr) + 0.3 * _bool_to_score(has_bullets) + freshness_bonus)


# Code quality: README presence + artifact count as (very) rough proxy; refine later with lint/tests.
def metric_code_quality(inp: Inputs) -> float:
    md = inp.metadata or {}
    text = inp.readme or ""
    readme_ok = len(text) > 200
    file_count = md.get("fileCount") or 0
    freshness = 1.0 if md.get("lastModified") else 0.0
    score = 0.4 * _bool_to_score(readme_ok) + 0.3 * min(1.0, file_count / 12.0) + 0.3 * freshness
    return _clamp01(score)


# ------------------------- Registry & Orchestrator ------------------------- #

MetricFn = Callable[[Inputs], float | Dict[str, float]]

# Order here does not matter; we’ll execute in parallel and then assemble.
METRICS: Dict[str, MetricFn] = {
    "ramp_up_time": metric_ramp_up_time,
    "bus_factor": metric_bus_factor,
    "performance_claims": metric_performance_claims,
    "license": metric_license,
    "size_score": metric_size_score,  # object
    "dataset_and_code_score": metric_dataset_and_code_score,
    "dataset_quality": metric_dataset_quality,
    "code_quality": metric_code_quality,
}

# Your team’s weights from the plan (sum to 1.0)
NET_WEIGHTS: Dict[str, float] = {
    "license": 0.15,
    "ramp_up_time": 0.15,
    "bus_factor": 0.12,
    "dataset_and_code_score": 0.11,
    "dataset_quality": 0.12,
    "code_quality": 0.12,
    "performance_claims": 0.12,
    "size_score": 0.11,  # we’ll average the hardware buckets into one scalar for the weighted sum
}


def _size_scalar(size_obj: Dict[str, float]) -> float:
    if not size_obj:
        return 0.0
    vals = [v for v in size_obj.values() if isinstance(v, (int, float))]
    return _clamp01(sum(vals) / max(1, len(vals)))


def _cpu_workers() -> int:
    try:
        import os as _os
        n = len(_os.sched_getaffinity(0))  # Linux-friendly
    except Exception:
        n = os.cpu_count() or 2
    # keep it modest; Phase-1 shouldn’t DOS the grader box
    return max(2, min(8, n))


def score_resource(resource: Resource) -> Dict[str, Any]:
    """
    Compute all metrics for a given Resource (usually a ModelResource).
    Returns a dict with every Table-1 field (latencies in ms as ints).
    """
    # Fetch metadata + README once here (I/O), then compute metrics on that snapshot.
    try:
        metadata = resource.fetchMetadata() or {}
    except Exception as e:
        LOG.debug(f"fetchMetadata error: {e}")
        metadata = {}

    try:
        readme = resource.fetchReadme()
    except Exception as e:
        LOG.debug(f"fetchReadme error: {e}")
        readme = None

    inp = Inputs(resource=resource, metadata=metadata, readme=readme)

    # Run metrics in parallel
    results: Dict[str, Tuple[float | Dict[str, float], int]] = {}
    with cf.ThreadPoolExecutor(max_workers=_cpu_workers()) as ex:
        fut_to_name = {
            ex.submit(_latency_wrapper, lambda f=f: f(inp)): name  # bind current f
            for name, f in METRICS.items()
        }
        for fut in cf.as_completed(fut_to_name):
            name = fut_to_name[fut]
            val, lat = fut.result()
            results[name] = (val, int(lat))

    # Assemble output record
    name = getattr(resource, "ref", None).name if getattr(resource, "ref", None) else None
    category = getattr(resource, "ref", None).category.name if getattr(resource, "ref", None) else "UNKNOWN"

    record: Dict[str, Any] = {
        "name": name or "",
        "category": category,  # MODEL / DATASET / CODE
    }

    # Fill scalar metrics + latencies
    for m in ("ramp_up_time", "bus_factor", "performance_claims", "license",
              "dataset_and_code_score", "dataset_quality", "code_quality"):
        val, lat = results.get(m, (0.0, 0))
        record[m] = _clamp01(val if isinstance(val, (int, float)) else 0.0)  # type: ignore[arg-type]
        record[f"{m}_latency"] = int(lat)

    # size_score is an object + latency
    size_val, size_lat = results.get("size_score", ({}, 0))
    size_obj: Dict[str, float] = size_val if isinstance(size_val, dict) else {}
    record["size_score"] = {k: _clamp01(v) for k, v in size_obj.items()}
    record["size_score_latency"] = int(size_lat)

    # Net score (weighted sum; size_score -> scalar by averaging buckets)
    t0 = _now_ms()
    net = 0.0
    for key, w in NET_WEIGHTS.items():
        if key == "size_score":
            net += w * _size_scalar(record["size_score"])
        else:
            net += w * float(record.get(key, 0.0))
    record["net_score"] = _clamp01(net)
    record["net_score_latency"] = max(0, _now_ms() - t0)

    return record


# ------------------------- Minimal demo (optional) ------------------------- #

if __name__ == "__main__":  # pragma: no cover
    # Tiny sanity check if you run this file directly.
    from URL_Fetcher import determineResource  # type: ignore

    demo_url = "https://huggingface.co/google/flan-t5-base"
    res = determineResource(demo_url)
    out = score_resource(res)
    # Do not print logs here; this is just for developers:
    import json as _json
    print(_json.dumps(out, indent=2))
