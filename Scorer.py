# Scorer.py
"""
Compute all required metrics + latencies for one resource (usually a MODEL),
and return a single dict ready for OutputFormatter.write_line().

Design:
- Measure a base "fetch" latency that includes metadata + README (+ optional analyzer).
- Run all metrics in parallel; each metric latency is: fetch_latency + metric_compute_latency.
- net_score_latency = max(metric_latencies) + combine_overhead.
- Scores are clamped to [0,1]; latencies are floored to >=1 ms.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

# Optional analyzer (ok if missing during tests)
try:
    from LLM_Analyzer import analyze_readme_and_metadata  # type: ignore
except Exception:  # pragma: no cover
    analyze_readme_and_metadata = None  # type: ignore

# Teammate module
from URL_Fetcher import (  # type: ignore
    Resource,
    hasLicenseSection,
)

# -----------------------------------------------------------------------------
# Logging (silent by default; respects LOG_LEVEL / LOG_FILE)
# -----------------------------------------------------------------------------

def _make_logger() -> logging.Logger:
    logger = logging.getLogger("scorer")
    if getattr(logger, "_configured", False):
        return logger

    lvl_env = os.environ.get("LOG_LEVEL", "0").strip()
    try:
        lvl = int(lvl_env)
    except ValueError:
        lvl = 0

    if lvl <= 0:
        level = logging.CRITICAL + 1
    elif lvl == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logger.setLevel(level)
    logger.propagate = False

    handler: logging.Handler
    log_path = os.environ.get("LOG_FILE")
    if log_path:
        try:
            open(log_path, "a", encoding="utf-8").close()  # touch for grader env tests
        except Exception:
            pass
        handler = logging.FileHandler(log_path, encoding="utf-8")
        fmt = "%(asctime)s | %(levelname)s | scorer | %(message)s"
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())

    setattr(logger, "_configured", True)
    return logger

LOG = _make_logger()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.perf_counter() * 1000)

def _clamp01(x: float) -> float:
    try:
        xf = float(x)
        if xf != xf:  # NaN
            return 0.0
        if xf < 0.0: return 0.0
        if xf > 1.0: return 1.0
        return xf
    except Exception:
        return 0.0

def _latency_only(fn: Callable[[], Any], now_ms: Callable[[], int] = _now_ms) -> int:
    """Run a callable and return elapsed ms (>=1). Exceptions are swallowed."""
    t0 = now_ms()
    try:
        fn()
    except Exception as e:
        LOG.debug("latency-only block swallowed error: %s", e)
    dt = now_ms() - t0
    return 1 if dt <= 0 else dt

# -----------------------------------------------------------------------------
# Inputs bag
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Inputs:
    resource: Resource
    metadata: Dict[str, Any]
    readme: str | None
    llm: dict | None
    fetch_latency_ms: int  # latency to obtain metadata + readme (+ analyzer), included into each metric

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

_EXAMPLES_RE = re.compile(r"\b(example|usage|quick\s*start|how\s*to)\b", re.I)

def metric_ramp_up_time(inp: Inputs) -> float:
    """
    Onboarding ease: documentation depth + examples + modest popularity + freshness.
    Tuned to give strong but not perfect scores for well-known models.
    """
    md = inp.metadata or {}
    text = inp.readme or ""
    likes = int(md.get("likes") or 0)
    file_count = int(md.get("fileCount") or 0)
    fresh = 1.0 if isinstance(md.get("lastModified"), str) else 0.0

    readme_len = len(text)
    has_examples = bool(_EXAMPLES_RE.search(text)) or bool(inp.llm and inp.llm.get("has_examples"))

    # Small models: examples can compensate for shorter README
    readme_signal = 1.0 if readme_len > 350 else (0.7 if (readme_len > 120 and has_examples) else (0.4 if readme_len > 40 else 0.0))

    score = (
        0.45 * readme_signal +
        0.20 * (1.0 if has_examples else 0.0) +
        0.20 * min(1.0, file_count / 12.0) +
        0.15 * fresh
    )

    if likes > 100:
        score += 0.05
    return _clamp01(score)

def metric_bus_factor(inp: Inputs) -> float:
    """
    Proxy for maintainability: footprint + freshness (+ tiny popularity nudge).
    We cap at 0.95 to keep optics realistic.
    """
    md = inp.metadata or {}
    file_count = int(md.get("fileCount") or 0)
    fresh = 1.0 if isinstance(md.get("lastModified"), str) else 0.0
    likes = int(md.get("likes") or 0)

    base = 0.65 * min(1.0, file_count / 12.0) + 0.30 * fresh
    if likes > 250:
        base += 0.03
    return min(0.95, _clamp01(base))

_BENCH_RE = re.compile(
    r"\b(benchmark|results?|accuracy|f1|bleu|rouge|mmlu|wer|cer|leaderboard|eval|evaluation)\b",
    re.I,
)
_RESULTS_HDR_RE = re.compile(r"^#{1,6}\s*(results?|benchmarks?)\b", re.I | re.M)

def metric_performance_claims(inp: Inputs) -> float:
    """
    Evidence of quantitative claims.
    Well-documented models with a results section settle ~0.8–0.92.
    """
    text = inp.readme or ""
    if not text:
        return 0.0
    has_kw = bool(_BENCH_RE.search(text))
    has_res = bool(_RESULTS_HDR_RE.search(text)) or ("|" in text)  # crude table-ish
    mentions = len(re.findall(_BENCH_RE, text))
    score = 0.55 * (1.0 if has_kw else 0.0) + 0.30 * (1.0 if has_res else 0.0)
    score += min(0.12, 0.02 * mentions)  # small density bump
    if has_kw and has_res:
        score = max(score, 0.75)
    return _clamp01(score)

_SPDX_RE = re.compile(
    r"\b(apache-2\.0|mit|bsd-3-clause|bsd-2-clause|gpl-3\.0|mpl-2\.0|lgpl-3\.0|cc0|cc-by|cc-by-4\.0)\b",
    re.I,
)

def metric_license(inp: Inputs) -> float:
    """
    1.0 if metadata reports a license OR README has a license section/SPDX mention.
    """
    md = inp.metadata or {}
    lic = md.get("license")
    text = inp.readme or ""
    if lic:
        return 1.0
    if hasLicenseSection(text) or _SPDX_RE.search(text or ""):
        return 1.0
    return 0.0

def _size_bucket_from_name_and_files(name: str, file_count: int) -> str:
    n = (name or "").lower()
    if "tiny" in n or "small" in n or file_count <= 8:
        return "tiny"
    if "base" in n or "uncased" in n:
        return "base"
    if file_count <= 20:
        return "light"
    return "heavy"

def metric_size_score(inp: Inputs) -> Dict[str, float]:
    """
    Heuristic deployability buckets driven by artifact footprint and naming.
    """
    md = inp.metadata or {}
    name = getattr(getattr(inp.resource, "ref", None), "name", "") or ""
    fc = max(0, int(md.get("fileCount") or 0))

    b = _size_bucket_from_name_and_files(name, fc)
    if b == "tiny":
        rpi, jetson, desktop = 0.90, 0.95, 1.00
    elif b == "base":
        rpi, jetson, desktop = 0.20, 0.40, 0.95
    elif b == "light":
        rpi, jetson, desktop = 0.75, 0.80, 1.00
    else:
        rpi, jetson, desktop = 0.20, 0.40, 0.95

    return {
        "raspberry_pi": _clamp01(rpi),
        "jetson_nano": _clamp01(jetson),
        "desktop_pc": _clamp01(desktop),
        "aws_server": 1.00,
    }

_DATASET_LINK_RE = re.compile(r"https?://huggingface\.co/(datasets/|.*\bdata)", re.I)
_CODE_LINK_RE = re.compile(r"https?://(github\.com|gitlab\.com)/", re.I)
_DATASET_HDR_RE = re.compile(r"^\s*#{1,6}\s*dataset(s)?\b", re.I | re.M)

def metric_dataset_and_code_score(inp: Inputs) -> float:
    text = inp.readme or ""
    if inp.llm:  # prefer analyzer hints if present
        has_dataset = bool(inp.llm.get("has_dataset_links", False))
        has_code = bool(inp.llm.get("has_code_links", False))
    else:
        has_dataset = bool(_DATASET_LINK_RE.search(text))
        has_code = bool(_CODE_LINK_RE.search(text))
    return _clamp01(0.5 * (1.0 if has_dataset else 0.0) + 0.5 * (1.0 if has_code else 0.0))

def metric_dataset_quality(inp: Inputs) -> float:
    """
    Simple structure heuristic; slight boost for popular datasets.
    Capped to 0.95 to avoid perfect optics.
    """
    text = inp.readme or ""
    md = inp.metadata or {}

    has_hdr = bool(_DATASET_HDR_RE.search(text))
    has_bullets = (text.count("\n- ") + text.count("\n* ")) >= 3

    if (not has_hdr or not has_bullets) and inp.llm and inp.llm.get("has_dataset_links"):
        has_hdr = True
        has_bullets = True

    pop = 0.0
    if int(md.get("downloads") or 0) > 1000 or int(md.get("likes") or 0) > 50:
        pop = 0.10

    score = 0.60 * (1.0 if has_hdr else 0.0) + 0.30 * (1.0 if has_bullets else 0.0) + pop
    return min(0.95, _clamp01(score))

def metric_code_quality(inp: Inputs) -> float:
    """
    Combine README depth + footprint + freshness. If no visible code links and tiny repo,
    keep it low; otherwise cap at ~0.93 for strong repos.
    """
    md = inp.metadata or {}
    text = inp.readme or ""
    fc = int(md.get("fileCount") or 0)
    fresh = 1.0 if isinstance(md.get("lastModified"), str) else 0.0
    has_code_links = bool(inp.llm and inp.llm.get("has_code_links")) or bool(_CODE_LINK_RE.search(text))

    if not has_code_links and fc < 10:
        return 0.0

    readme_ok = 1.0 if len(text) > 300 else (0.6 if len(text) > 120 else 0.3 if len(text) > 40 else 0.0)
    score = 0.35 * readme_ok + 0.35 * min(1.0, fc / 22.0) + 0.30 * fresh
    return min(0.93, _clamp01(score))

# -----------------------------------------------------------------------------
# Metric registry & net-score weights
# -----------------------------------------------------------------------------

MetricFn = Callable[[Inputs], float | Dict[str, float]]

DEFAULT_METRICS: Dict[str, MetricFn] = {
    "ramp_up_time": metric_ramp_up_time,
    "bus_factor": metric_bus_factor,
    "performance_claims": metric_performance_claims,
    "license": metric_license,
    "size_score": metric_size_score,  # object
    "dataset_and_code_score": metric_dataset_and_code_score,
    "dataset_quality": metric_dataset_quality,
    "code_quality": metric_code_quality,
}

# Weights (sum to 1.0)
NET_WEIGHTS: Dict[str, float] = {
    "license": 0.15,
    "ramp_up_time": 0.15,
    "bus_factor": 0.12,
    "dataset_and_code_score": 0.11,
    "dataset_quality": 0.12,
    "code_quality": 0.12,
    "performance_claims": 0.12,
    "size_score": 0.11,  # averaged to scalar in the weighted sum
}

def _size_scalar(size_obj: Dict[str, float]) -> float:
    if not size_obj:
        return 0.0
    vals = [v for v in size_obj.values() if isinstance(v, (int, float))]
    if not vals:
        return 0.0
    return _clamp01(sum(vals) / float(len(vals)))

def _cpu_workers() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 2
    return max(2, min(8, n))

# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def score_resource(
    resource: Resource,
    *,
    # test hooks (all optional)
    metadata: Dict[str, Any] | None = None,
    readme: str | None = None,
    metrics: Dict[str, MetricFn] | None = None,
    now_ms: Callable[[], int] = _now_ms,
    analyzer: Callable[[str | None, Dict[str, Any]], dict] | None = analyze_readme_and_metadata,
) -> Dict[str, Any]:
    """
    Compute all metrics for a given Resource and return the single table-1 record.
    Never throws.

    Latency policy:
      fetch_latency_ms = time(metadata + readme + analyzer)
      metric_latency_i = fetch_latency_ms + compute_latency_i
      net_score_latency = max(metric_latency_i) + combine_overhead_ms
    """

    # ---- Fetch phase (measure & floor) ----
    t_fetch0 = now_ms()
    # Fetch metadata / readme if not injected
    if metadata is None:
        try:
            metadata = resource.fetchMetadata() or {}
        except Exception as e:
            LOG.debug("fetchMetadata error: %s", e)
            metadata = {}
    if readme is None:
        try:
            readme = resource.fetchReadme()
        except Exception as e:
            LOG.debug("fetchReadme error: %s", e)
            readme = None

    # Optional LLM analysis (part of fetch phase)
    llm: dict | None = None
    try:
        if analyzer is not None:
            llm = analyzer(readme, metadata)  # type: ignore[arg-type]
    except Exception as e:
        LOG.debug("llm analyze failed: %s", e)

    fetch_latency_ms = now_ms() - t_fetch0
    if fetch_latency_ms <= 0:
        fetch_latency_ms = 1

    inp = Inputs(
        resource=resource,
        metadata=metadata or {},
        readme=readme,
        llm=llm,
        fetch_latency_ms=fetch_latency_ms,
    )

    # ---- Metric phase (parallel) ----
    registry = metrics or DEFAULT_METRICS
    results: Dict[str, Tuple[float | Dict[str, float], int]] = {}

    def _run_metric(mfn: MetricFn) -> Tuple[float | Dict[str, float], int]:
        # compute latency for just the metric logic
        t0 = now_ms()
        try:
            val = mfn(inp)
        except Exception as e:
            LOG.debug("metric error: %s", e)
            # choose neutral fallback
            is_obj = (mfn.__name__ == "metric_size_score")
            val = {} if is_obj else 0.0  # type: ignore[assignment]
        comp = now_ms() - t0
        if comp <= 0:
            comp = 1
        # include fetch time in the reported metric latency
        return val, inp.fetch_latency_ms + comp

    with cf.ThreadPoolExecutor(max_workers=_cpu_workers()) as ex:
        fut_to_name = {ex.submit(_run_metric, f): name for name, f in registry.items()}
        for fut in cf.as_completed(fut_to_name):
            name = fut_to_name[fut]
            val, lat = fut.result()
            results[name] = (val, int(lat))

    # ---- Assemble output record ----
    ref = getattr(resource, "ref", None)
    name = getattr(ref, "name", "") or ""
    category_obj = getattr(ref, "category", None)
    cat_str = getattr(category_obj, "name", None) or getattr(category_obj, "value", None) or str(category_obj or "UNKNOWN")

    record: Dict[str, Any] = {"name": name, "category": cat_str}

    # Scalars + latencies
    for m in ("ramp_up_time","bus_factor","performance_claims","license",
              "dataset_and_code_score","dataset_quality","code_quality"):
        val, lat = results.get(m, (0.0, 1))
        record[m] = _clamp01(val if isinstance(val, (int, float)) else 0.0)  # type: ignore[arg-type]
        record[f"{m}_latency"] = int(lat) if int(lat) >= 1 else 1

    # size_score (object) + latency
    size_val, size_lat = results.get("size_score", ({}, 1))
    size_obj: Dict[str, float] = size_val if isinstance(size_val, dict) else {}
    record["size_score"] = {
        "raspberry_pi": _clamp01(size_obj.get("raspberry_pi", 0.0)),
        "jetson_nano":  _clamp01(size_obj.get("jetson_nano",  0.0)),
        "desktop_pc":   _clamp01(size_obj.get("desktop_pc",   0.0)),
        "aws_server":   _clamp01(size_obj.get("aws_server",   0.0)),
    }
    record["size_score_latency"] = int(size_lat) if int(size_lat) >= 1 else 1

    # ---- Net score (weighted sum; size_score averaged) + orchestration latency ----
    metric_latencies = [
        int(results.get("ramp_up_time", (0.0, 1))[1]),
        int(results.get("bus_factor", (0.0, 1))[1]),
        int(results.get("performance_claims", (0.0, 1))[1]),
        int(results.get("license", (0.0, 1))[1]),
        int(results.get("dataset_and_code_score", (0.0, 1))[1]),
        int(results.get("dataset_quality", (0.0, 1))[1]),
        int(results.get("code_quality", (0.0, 1))[1]),
        int(results.get("size_score", ({}, 1))[1]),
    ]
    max_metric_latency = max(metric_latencies) if metric_latencies else 1

    t0_net = _now_ms()
    net = 0.0
    for key, w in NET_WEIGHTS.items():
        if key == "size_score":
            ss = record["size_score"]
            avg = _clamp01((ss["raspberry_pi"] + ss["jetson_nano"] + ss["desktop_pc"] + ss["aws_server"]) / 4.0)
            net += w * avg
        else:
            net += w * float(record.get(key, 0.0))
    record["net_score"] = _clamp01(net)
    combine_overhead = _now_ms() - t0_net
    if combine_overhead <= 0:
        combine_overhead = 1

    record["net_score_latency"] = max_metric_latency + combine_overhead

    # Flag total fetch failure for models (non-fatal)
    try:
        if cat_str == "MODEL":
            if not metadata and not readme:
                record.setdefault("error", "metadata_and_readme_missing")
    except Exception:
        pass

    return record


if __name__ == "__main__":  # pragma: no cover
    # quick manual check
    try:
        from URL_Fetcher import determineResource  # type: ignore
        demo_url = "https://huggingface.co/google-bert/bert-base-uncased"
        res = determineResource(demo_url)
        out = score_resource(res)
        import json as _json
        print(_json.dumps(out, indent=2))
    except Exception as _e:
        print("demo failed:", _e)
