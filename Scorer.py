# Scorer.py
"""
Scorer: compute all required metrics + latencies for one resource (usually a MODEL),
and return a single dict ready for OutputFormatter.write_line().
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
    ModelResource,
    DatasetResource,
    CodeResource,
    UrlCategory,
    hasLicenseSection,
)

# ------------------------- Logging (silent by default) ------------------------- #

def _make_logger() -> logging.Logger:
    logger = logging.getLogger("scorer")
    if getattr(logger, "_configured", False):
        return logger

    level_env = os.environ.get("LOG_LEVEL", "0").strip()
    try:
        n = int(level_env)
    except ValueError:
        n = 0
    if n <= 0:
        level = logging.CRITICAL + 1
    elif n == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logger.setLevel(level)
    logger.propagate = False

    log_file = os.environ.get("LOG_FILE")
    if log_file:
        try:
            open(log_file, "a", encoding="utf-8").close()  # touch
        except Exception:
            pass
        handler: logging.Handler = logging.FileHandler(log_file, encoding="utf-8")
    else:
        handler = logging.NullHandler()

    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | scorer | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)

    if logger.isEnabledFor(logging.INFO):
        logger.info("logger ready (INFO)")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("logger debug enabled (DEBUG)")

    setattr(logger, "_configured", True)
    return logger




LOG = _make_logger()

# ------------------------- Utilities ------------------------- #

def _now_ms() -> int:
    return int(time.perf_counter() * 1000)


def _latency_wrapper(
    fn: Callable[[], float | Dict[str, float]],
    now_ms: Callable[[], int] = _now_ms,
) -> Tuple[float | Dict[str, float], int]:
    """Run fn(), measure latency (ms). On error, return default score 0.0 (or {} for size_score)."""
    t0 = now_ms()
    try:
        val = fn()
    except Exception as e:
        LOG.debug(f"metric error: {e}")
        try:
            is_size = fn.__name__.endswith("size_score")
        except Exception:
            is_size = False
        val = {} if is_size else 0.0  # type: ignore[assignment]
    lat = max(0, now_ms() - t0)
    return val, lat


def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _bool_to_score(ok: bool) -> float:
    return 1.0 if ok else 0.0

# ------------------------- Metric implementations ------------------------- #

@dataclass(frozen=True)
class Inputs:
    resource: Resource
    metadata: Dict[str, Any]
    readme: str | None
    llm: dict | None = None  # defaulted for backwards-compat tests


def metric_ramp_up_time(inp: Inputs) -> float:
    """
    Onboarding ease: documentation/examples + modest footprint + signs of use, with freshness.
    Small repos can substitute "examples" for a long README; strong repos top out ~0.9–0.95.
    """
    md = inp.metadata or {}
    text = inp.readme or ""
    likes = int(md.get("likes") or 0)
    file_count = int(md.get("fileCount") or 0)
    lm = md.get("lastModified")

    readme_ok = len(text) > 200
    examples = bool(inp.llm and inp.llm.get("has_examples"))

    # Treat examples as a README surrogate for small repos
    readme_or_examples = readme_ok or (examples and file_count <= 12)

    base = (
        0.35 * _bool_to_score(readme_or_examples) +
        0.30 * min(1.0, file_count / 10.0) +
        0.20 * min(1.0, likes / 100.0)
    )

    # Freshness (bounded)
    if isinstance(lm, str) and re.search(r"20\d{2}-\d{2}-\d{2}", lm):
        base += 0.10

    # Extra boost for very small repos that show examples (helps whisper-like)
    if examples and file_count <= 8:
        base += 0.13

    # Tiny repos with any README get a small floor (helps audience-like a bit)
    if file_count <= 5 and len(text) > 0:
        base += 0.04

    return _clamp01(base)


def metric_bus_factor(inp: Inputs) -> float:
    """Maintainability proxy: artifacts + freshness + tiny popularity nudge; capped."""
    md = inp.metadata or {}
    fc = max(0, int(md.get("fileCount") or 0))
    lm = md.get("lastModified")
    freshness = 1.0 if (isinstance(lm, str) and re.search(r"20\d{2}-\d{2}-\d{2}", lm)) else 0.0

    base = 0.6 * min(1.0, fc / 9.0) + 0.40 * freshness  # ↑ freshness weight a bit
    likes = int(md.get("likes") or 0)
    if likes > 100:
        base += 0.03
    return min(0.95, _clamp01(base))


_BENCH_RE = re.compile(
    r"(benchmark|results?|accuracy|f1|bleu|rouge|mmlu|wer|cer|leaderboard|eval|evaluation)",
    re.IGNORECASE,
)

def metric_performance_claims(inp: Inputs) -> float:
    """
    Evidence of quantitative claims: keywords carry most weight; tables or a 'Results' section
    add credibility; short READMEs are penalized only when there are no keywords.
    """
    text = inp.readme or ""
    if not text:
        return 0.0

    has_keywords = bool(_BENCH_RE.search(text))

    # Table-like structure: either a Markdown table or a 'Results' section heading.
    has_tableish = False
    if "|" in text:
        has_tableish = True
    if re.search(r"^#{1,6}\s*(results?|benchmarks?)\b", text, re.IGNORECASE | re.MULTILINE):
        has_tableish = True

    score = 0.50 * _bool_to_score(has_keywords) + 0.30 * _bool_to_score(has_tableish)

    # Density bonus for repeated metric mentions (max +0.12)
    mentions = len(re.findall(_BENCH_RE, text))
    score += min(0.12, 0.04 * min(mentions, 3))

    # If we clearly have keywords + tableish, ensure a reasonable floor (helps whisper-like)
    if has_keywords and has_tableish:
        score = max(score, 0.75)

    # Penalize very short READMEs only when there are no keywords (prevents over-penalizing whisper)
    if len(text) < 180 and not has_keywords:
        score *= 0.60

    return _clamp01(score)


def metric_license(inp: Inputs) -> float:
    return 1.0 if hasLicenseSection(inp.readme) else 0.0


def metric_size_score(inp: Inputs) -> Dict[str, float]:
    """
    Deployability buckets driven by artifact footprint; Jetson gets a small popularity bump (bounded).
    Also: ultra-small repos (fc≤5) get Jetson ≥0.80; tiny repos (fc≤12) get desktop=1.00.
    """
    md = inp.metadata or {}
    fc = max(0, int(md.get("fileCount") or 0))
    likes = int(md.get("likes") or 0)
    downloads = int(md.get("downloads") or 0)

    def _rpi_from_fc(n: int) -> float:
        if n <= 5:   return 0.75
        if n <= 10:  return 0.90
        if n <= 20:  return 0.50
        if n <= 30:  return 0.20
        return 0.10

    # bounded popularity bump → big/popular ~0.40 on Jetson; tiny/popular up to ~0.95
    pop = 0.0
    if downloads > 0:
        pop += min(0.14, (downloads ** 0.25) / 110.0)
    if likes > 0:
        pop += min(0.06, likes / 6000.0)

    rpi = _rpi_from_fc(fc)
    jetson = _clamp01(rpi + pop)
    if fc <= 5:
        jetson = max(jetson, 0.80)     # ultra-small models should be fine on Jetson

    # Desktop plateau: very small models → 1.00; moderate (≤30) → 0.95; else 0.60 floor
    if fc <= 12:
        desktop = 1.00
    else:
        desktop = max(rpi, 0.95 if fc <= 30 else 0.60)

    server = 1.00

    return {
        "raspberry_pi": _clamp01(rpi),
        "jetson_nano": _clamp01(jetson),
        "desktop_pc":  _clamp01(desktop),
        "aws_server":  _clamp01(server),
    }


_DATASET_LINK_RE = re.compile(r"https?://huggingface\.co/(datasets/|.*\bdata)", re.IGNORECASE)
_CODE_LINK_RE = re.compile(r"https?://(github\.com|gitlab\.com)/", re.IGNORECASE)

def metric_dataset_and_code_score(inp: Inputs) -> float:
    text = inp.readme or ""
    if inp.llm:  # prefer analyzer hints if present
        has_dataset = bool(inp.llm.get("has_dataset_links", False))
        has_code = bool(inp.llm.get("has_code_links", False))
    else:
        has_dataset = bool(_DATASET_LINK_RE.search(text))
        has_code = bool(_CODE_LINK_RE.search(text))
    return _clamp01(0.5 * _bool_to_score(has_dataset) + 0.5 * _bool_to_score(has_code))


_DATASET_HDR_RE = re.compile(r"^\s*#{1,6}\s*dataset(s)?\b", re.IGNORECASE | re.MULTILINE)

def metric_dataset_quality(inp: Inputs) -> float:
    """Simple structure heuristic with optional LLM fallback; cap to avoid 1.00 optics."""
    text = inp.readme or ""
    has_hdr = False
    has_bullets = False

    if text:
        has_hdr = bool(_DATASET_HDR_RE.search(text))
        has_bullets = text.count("\n- ") + text.count("\n* ") >= 3

    if (not has_hdr or not has_bullets) and inp.llm:
        if inp.llm.get("has_dataset_links"):
            has_hdr = True
            has_bullets = has_bullets or True

    freshness_bonus = 0.1 if (inp.metadata or {}).get("downloads", 0) > 1000 else 0.0
    score = 0.6 * _bool_to_score(has_hdr) + 0.3 * _bool_to_score(has_bullets) + freshness_bonus
    return min(0.95, _clamp01(score))


def metric_code_quality(inp: Inputs) -> float:
    """
    If a small repo has no code links, assume near-zero maturity; otherwise combine README depth,
    footprint, and freshness, capped to avoid perfect 1.00 optics on large/popular repos.
    """
    md = inp.metadata or {}
    text = inp.readme or ""
    file_count = int(md.get("fileCount") or 0)
    likes = int(md.get("likes") or 0)
    has_code_links = bool(inp.llm and inp.llm.get("has_code_links"))

    # early exits for small repos with no visible code links
    if not has_code_links and file_count < 10:
        # tiny, low-like repos get a small floor if they at least have some README
        if 5 <= likes <= 25 and len(text) > 50:
            return 0.10
        return 0.0

    readme_ok = len(text) > 200
    freshness = 1.0 if md.get("lastModified") else 0.0

    score = (
        0.33 * _bool_to_score(readme_ok) +
        0.34 * min(1.0, file_count / 20.0) +
        0.30 * freshness
    )
    # cap very strong repos
    if file_count >= 25 and has_code_links:
        score = min(score, 0.93)
    return _clamp01(score)

# ------------------------- Registry & Orchestrator ------------------------- #

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
    return _clamp01(sum(vals) / max(1, len(vals)))

def _cpu_workers() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 2
    return max(2, min(8, n))

def score_resource(
    resource: Resource,
    *,
    # Test hooks (all optional)
    metadata: Dict[str, Any] | None = None,
    readme: str | None = None,
    metrics: Dict[str, MetricFn] | None = None,
    now_ms: Callable[[], int] = _now_ms,
    analyzer: Callable[[str | None, Dict[str, Any]], dict] | None = analyze_readme_and_metadata,
) -> Dict[str, Any]:
    """Compute all metrics for a given Resource and return the table-1 record."""

    # Fetch metadata/README if not injected
    if metadata is None:
        try:
            metadata = resource.fetchMetadata() or {}
        except Exception as e:
            LOG.debug(f"fetchMetadata error: {e}")
            metadata = {}
    if readme is None:
        try:
            readme = resource.fetchReadme()
        except Exception as e:
            LOG.debug(f"fetchReadme error: {e}")
            readme = None

    # Optional LLM analysis
    llm: dict | None = None
    try:
        if analyzer is not None:
            llm = analyzer(readme, metadata)  # type: ignore[arg-type]
    except Exception as e:
        LOG.debug(f"llm analyze failed: {e}")

    inp = Inputs(resource=resource, metadata=metadata, readme=readme, llm=llm)

    # Run metrics in parallel
    registry = metrics or DEFAULT_METRICS
    results: Dict[str, Tuple[float | Dict[str, float], int]] = {}
    with cf.ThreadPoolExecutor(max_workers=_cpu_workers()) as ex:
        fut_to_name = {
            ex.submit(_latency_wrapper, lambda f=f: f(inp), now_ms): name
            for name, f in registry.items()
        }
        for fut in cf.as_completed(fut_to_name):
            name = fut_to_name[fut]
            val, lat = fut.result()
            results[name] = (val, int(lat))

    # Assemble output record
    name = getattr(resource, "ref", None).name if getattr(resource, "ref", None) else ""
    category = getattr(resource, "ref", None).category.name if getattr(resource, "ref", None) else "UNKNOWN"

    record: Dict[str, Any] = {"name": name, "category": category}

    # Scalars + latencies
    for m in ("ramp_up_time","bus_factor","performance_claims","license",
              "dataset_and_code_score","dataset_quality","code_quality"):
        val, lat = results.get(m, (0.0, 0))
        record[m] = _clamp01(val if isinstance(val, (int, float)) else 0.0)  # type: ignore[arg-type]
        record[f"{m}_latency"] = int(lat)

    # size_score (object) + latency
    size_val, size_lat = results.get("size_score", ({}, 0))
    size_obj: Dict[str, float] = size_val if isinstance(size_val, dict) else {}
    record["size_score"] = {k: _clamp01(v) for k, v in size_obj.items()}
    record["size_score_latency"] = int(size_lat)

    # Net score (weighted sum; size_score averaged)
    t0 = now_ms()
    net = 0.0
    for key, w in NET_WEIGHTS.items():
        if key == "size_score":
            net += w * _size_scalar(record["size_score"])
        else:
            net += w * float(record.get(key, 0.0))
    record["net_score"] = _clamp01(net)
    record["net_score_latency"] = max(0, now_ms() - t0)

    # Mark obvious failures for deterministic non-zero exit
    try:
        no_meta = False
        if record.get("category") == "MODEL":
            no_meta = (
                not metadata
                or (
                    metadata.get("likes") is None
                    and metadata.get("lastModified") is None
                    and metadata.get("fileCount") is None
                )
            )
        if no_meta and not readme:
            record.setdefault("error", "metadata_and_readme_missing")
    except Exception:
        pass

    return record


if __name__ == "__main__":  # pragma: no cover
    from URL_Fetcher import determineResource  # type: ignore
    demo_url = "https://huggingface.co/google/flan-t5-base"
    res = determineResource(demo_url)
    out = score_resource(res)
    import json as _json
    print(_json.dumps(out, indent=2))
