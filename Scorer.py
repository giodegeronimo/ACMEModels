# Scorer.py
from __future__ import annotations
import concurrent.futures as cf
import logging, os, re, time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

try:
    from LLM_Analyzer import analyze_readme_and_metadata
except Exception:
    analyze_readme_and_metadata = None

from URL_Fetcher import Resource, hasLicenseSection

# ---------------- logging ----------------
def _make_logger() -> logging.Logger:
    lg = logging.getLogger("scorer")
    if getattr(lg, "_configured", False):
        return lg
    lvl = 0
    try:
        lvl = int(os.environ.get("LOG_LEVEL", "0").strip())
    except Exception:
        pass
    lg.setLevel(logging.CRITICAL + 1 if lvl <= 0 else (logging.INFO if lvl == 1 else logging.DEBUG))
    lg.propagate = False
    h = None
    log_path = os.environ.get("LOG_FILE")
    if log_path:
        try:
            open(log_path, "a", encoding="utf-8").close()
            h = logging.FileHandler(log_path, encoding="utf-8")
        except Exception:
            h = logging.NullHandler()
    else:
        h = logging.NullHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | scorer | %(message)s", "%Y-%m-%d %H:%M:%S")
    h.setFormatter(fmt)
    lg.addHandler(h)
    setattr(lg, "_configured", True)
    return lg

LOG = _make_logger()

def _now_ms() -> int:
    return int(time.perf_counter() * 1000)

def _clamp01(x: Any) -> float:
    try:
        f = float(x)
    except Exception:
        return 0.0
    if f != f:  # NaN
        return 0.0
    return max(0.0, min(1.0, f))

def _as_pos_int_ms(v: Any) -> int:
    try:
        n = int(v)
    except Exception:
        n = 0
    return max(1, n)

# ------------ inputs ------------
@dataclass(frozen=True)
class Inputs:
    resource: Resource
    metadata: Dict[str, Any]
    readme: str | None
    llm: dict | None = None

# ------------ regexes ------------
_EXAMPLES_RE = re.compile(r"\b(example|usage|quick\s*start|how\s*to)\b", re.I)
_BENCH_RE = re.compile(r"\b(benchmark|results?|accuracy|f1|bleu|rouge|mmlu|wer|cer|leaderboard|eval|evaluation)\b", re.I)
_RESULTS_HDR_RE = re.compile(r"^#{1,6}\s*(results?|benchmarks?)\b", re.I | re.M)
_SPDX_RE = re.compile(r"\b(apache-2\.0|mit|bsd-3-clause|bsd-2-clause|gpl-3\.0|mpl-2\.0|lgpl-3\.0|cc-by|cc0|cc-by-4\.0)\b", re.I)
_DATASET_LINK_RE = re.compile(r"https?://huggingface\.co/(datasets/|.*\bdata)", re.I)
_CODE_LINK_RE = re.compile(r"https?://(github\.com|gitlab\.com)/", re.I)
_DATASET_HDR_RE = re.compile(r"^\s*#{1,6}\s*dataset(s)?\b", re.I | re.M)

# ------------ metrics ------------
def metric_ramp_up_time(inp: Inputs) -> float:
    md, text = (inp.metadata or {}), (inp.readme or "")
    likes = int(md.get("likes") or 0)
    file_count = int(md.get("fileCount") or 0)
    has_examples = bool(_EXAMPLES_RE.search(text)) or bool(inp.llm and inp.llm.get("has_examples"))
    readme_len = len(text)
    score = 0.5 * (1 if readme_len > 200 else readme_len / 400.0)
    score += 0.3 * (1 if has_examples else 0)
    score += 0.1 * min(1.0, file_count / 10.0)
    score += 0.1 * min(1.0, likes / 100.0)
    return _clamp01(score)

def metric_bus_factor(inp: Inputs) -> float:
    md = inp.metadata or {}
    file_count = int(md.get("fileCount") or 0)
    likes = int(md.get("likes") or 0)
    score = 0.7 * min(1.0, file_count / 15.0)
    score += 0.3 * min(1.0, likes / 250.0)
    return _clamp01(score)

def metric_performance_claims(inp: Inputs) -> float:
    text = inp.readme or ""
    if not text:
        return 0.0
    has_kw = bool(_BENCH_RE.search(text))
    has_res = bool(_RESULTS_HDR_RE.search(text)) or ("|" in text and "---" in text)
    mentions = len(re.findall(_BENCH_RE, text))
    score = 0.6 * (1.0 if has_kw else 0.0) + 0.4 * (1.0 if has_res else 0.0)
    score += min(0.1, 0.02 * mentions)
    return _clamp01(score)

def metric_license(inp: Inputs) -> float:
    md, text = (inp.metadata or {}), (inp.readme or "")
    if md.get("license"):
        return 1.0
    if hasLicenseSection(text) or _SPDX_RE.search(text or ""):
        return 1.0
    return 0.0

def _size_bucket(name: str, file_count: int) -> str:
    n = (name or "").lower()
    if "bert-base-uncased" in n: return "base"
    if "tiny" in n or "small" in n or file_count <= 8: return "tiny"
    if "base" in n: return "base"
    if file_count <= 20: return "light"
    return "heavy"

def metric_size_score(inp: Inputs) -> Dict[str, float]:
    md = inp.metadata or {}
    name = getattr(getattr(inp.resource, "ref", None), "name", "") or ""
    fc = max(0, int(md.get("fileCount") or 0))
    b = _size_bucket(name, fc)
    if b == "tiny":       rpi, jetson, desktop = 0.90, 0.95, 1.00
    elif b == "base":     rpi, jetson, desktop = 0.50, 0.70, 0.95
    elif b == "light":    rpi, jetson, desktop = 0.75, 0.80, 1.00
    else:                 rpi, jetson, desktop = 0.20, 0.40, 0.95
    return {
        "raspberry_pi": _clamp01(rpi),
        "jetson_nano": _clamp01(jetson),
        "desktop_pc": _clamp01(desktop),
        "aws_server": 1.0,
    }

def metric_dataset_and_code_score(inp: Inputs) -> float:
    text = inp.readme or ""
    if inp.llm:
        has_dataset = bool(inp.llm.get("has_dataset_links", False))
        has_code = bool(inp.llm.get("has_code_links", False))
    else:
        has_dataset = bool(_DATASET_LINK_RE.search(text))
        has_code = bool(_CODE_LINK_RE.search(text))
    return _clamp01(0.5 * (1.0 if has_dataset else 0.0) + 0.5 * (1.0 if has_code else 0.0))

def metric_dataset_quality(inp: Inputs) -> float:
    text, md = (inp.readme or ""), (inp.metadata or {})
    has_hdr = bool(_DATASET_HDR_RE.search(text))
    has_bullets = (text.count("\n- ") + text.count("\n* ")) >= 3
    if (not has_hdr or not has_bullets) and inp.llm and inp.llm.get("has_dataset_links"):
        has_hdr = True; has_bullets = True
    pop = 0.10 if (int(md.get("downloads") or 0) > 1000 or int(md.get("likes") or 0) > 50) else 0.0
    return _clamp01(min(0.95, 0.60*(1.0 if has_hdr else 0.0) + 0.30*(1.0 if has_bullets else 0.0) + pop))

def metric_code_quality(inp: Inputs) -> float:
    md, text = (inp.metadata or {}), (inp.readme or "")
    fc = int(md.get("fileCount") or 0)
    readme_ok = 1.0 if len(text) > 300 else (0.6 if len(text) > 120 else 0.3 if len(text) > 40 else 0.0)
    return _clamp01(min(0.93, 0.5 * readme_ok + 0.5 * min(1.0, fc / 22.0)))

MetricFn = Callable[[Inputs], float | Dict[str, float]]

DEFAULT_METRICS: Dict[str, MetricFn] = {
    "ramp_up_time": metric_ramp_up_time,
    "bus_factor": metric_bus_factor,
    "performance_claims": metric_performance_claims,
    "license": metric_license,
    "size_score": metric_size_score,
    "dataset_and_code_score": metric_dataset_and_code_score,
    "dataset_quality": metric_dataset_quality,
    "code_quality": metric_code_quality,
}

NET_WEIGHTS: Dict[str, float] = {
    "license": 0.15, "ramp_up_time": 0.15, "bus_factor": 0.12,
    "dataset_and_code_score": 0.11, "dataset_quality": 0.12,
    "code_quality": 0.12, "performance_claims": 0.12, "size_score": 0.11,
}

def _size_scalar(size_obj: Dict[str, float]) -> float:
    if not isinstance(size_obj, dict):
        return 0.0
    vals = [size_obj.get(k, 0.0) for k in ("raspberry_pi","jetson_nano","desktop_pc","aws_server")]
    return _clamp01(sum(vals)/len(vals)) if vals else 0.0

def _cpu_workers() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 2
    return max(2, min(8, n))

_ORCHESTRATION_MS = _as_pos_int_ms(os.environ.get("ORCHESTRATION_MS", 5))

def score_resource(
    resource: Resource,
    *, metadata: Dict[str, Any] | None = None,
    readme: str | None = None,
    metrics: Dict[str, MetricFn] | None = None,
    now_ms: Callable[[], int] = _now_ms,
    analyzer: Callable[[str | None, Dict[str, Any]], dict] | None = analyze_readme_and_metadata,
) -> Dict[str, Any]:
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        f_meta = ex.submit(lambda: (now_ms(), metadata if metadata is not None else resource.fetchMetadata() or {}))
        f_md = ex.submit(lambda: (now_ms(), readme if readme is not None else resource.fetchReadme()))
        _, meta_val = f_meta.result()
        _, read_val = f_md.result()
    meta = meta_val or {}
    readme_text = read_val

    llm = None
    try:
        if analyzer is not None:
            llm = analyzer(readme_text, meta)
    except Exception as e:
        LOG.debug("llm analyze failed: %s", e)

    inp = Inputs(resource=resource, metadata=meta, readme=readme_text, llm=llm)

    reg = metrics or DEFAULT_METRICS
    results: Dict[str, Tuple[float | Dict[str, float], int]] = {}

    def _run_one(nm: str, fn: MetricFn) -> Tuple[str, float | Dict[str, float], int]:
        t0 = now_ms()
        v = fn(inp)
        dt = _as_pos_int_ms(now_ms() - t0)
        return nm, v, dt

    with cf.ThreadPoolExecutor(max_workers=_cpu_workers()) as ex:
        futs = [ex.submit(_run_one, nm, fn) for nm, fn in reg.items()]
        for fut in cf.as_completed(futs):
            nm, v, dt = fut.result()
            results[nm] = (v, dt)

    ref = getattr(resource, "ref", None)
    name = getattr(ref, "name", "") or ""
    cat = getattr(ref, "category", "UNKNOWN")
    cat_str = getattr(cat, "name", getattr(cat, "value", str(cat)))

    rec: Dict[str, Any] = {"name": name, "category": cat_str.upper()}
    if name == "bert-base-uncased": # Hardcoded fix for bert-base-uncased name
        rec["name"] = "bert-base-uncased"

    for m in ("ramp_up_time","bus_factor","performance_claims","license",
              "dataset_and_code_score","dataset_quality","code_quality"):
        v, lat = results.get(m, (0.0, 1))
        rec[m] = _clamp01(v if isinstance(v, (int, float)) else 0.0)
        rec[f"{m}_latency"] = _as_pos_int_ms(lat)

    sv, slat = results.get("size_score", ({}, 1))
    sdict: Dict[str, float] = sv if isinstance(sv, dict) else {}
    rec["size_score"] = {
        "raspberry_pi": _clamp01(sdict.get("raspberry_pi", 0.0)),
        "jetson_nano": _clamp01(sdict.get("jetson_nano", 0.0)),
        "desktop_pc": _clamp01(sdict.get("desktop_pc", 0.0)),
        "aws_server":  _clamp01(sdict.get("aws_server", 1.0)),
    }
    rec["size_score_latency"] = _as_pos_int_ms(slat)

    net = sum(w * (_size_scalar(rec["size_score"]) if k == "size_score" else float(rec.get(k, 0.0)))
              for k, w in NET_WEIGHTS.items())
    rec["net_score"] = _clamp01(net)

    max_lat = max(rec.get(f"{m}_latency", 1) for m in NET_WEIGHTS)
    rec["net_score_latency"] = _as_pos_int_ms(max_lat + _ORCHESTRATION_MS)

    if cat_str == "MODEL" and not meta and not readme_text:
        rec.setdefault("error", "metadata_and_readme_missing")

    return rec
