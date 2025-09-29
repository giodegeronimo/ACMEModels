# Scorer.py
from __future__ import annotations
import concurrent.futures as cf
import hashlib, logging, os, re, time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

try:
    from LLM_Analyzer import analyze_readme_and_metadata  # type: ignore
except Exception:  # pragma: no cover
    analyze_readme_and_metadata = None  # type: ignore

from URL_Fetcher import Resource, hasLicenseSection  # type: ignore

def _make_logger() -> logging.Logger:
    lg = logging.getLogger("scorer")
    if getattr(lg, "_configured", False):
        return lg
    lvl = 0
    try: lvl = int(os.environ.get("LOG_LEVEL","0").strip())
    except Exception: lvl = 0
    lg.setLevel(logging.CRITICAL + 1 if lvl <= 0 else logging.INFO if lvl == 1 else logging.DEBUG)
    lg.propagate = False
    hp = os.environ.get("LOG_FILE")
    if hp:
        try: open(hp, "a", encoding="utf-8").close()
        except Exception: pass
        fh = logging.FileHandler(hp, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | scorer | %(message)s","%Y-%m-%d %H:%M:%S"))
        lg.addHandler(fh)
    else:
        lg.addHandler(logging.NullHandler())
    setattr(lg, "_configured", True)
    return lg

LOG = _make_logger()

def _now_ms() -> int: return int(time.perf_counter()*1000)
def _clamp01(x: float) -> float:
    try:
        f = float(x)
        return 0.0 if f < 0 else 1.0 if f > 1 else f
    except Exception:
        return 0.0

def _stable_hash_int(s: str) -> int:
    return int(hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:8], 16)

def _get_net_latency_target() -> int:
    try: tgt = int(os.environ.get("NET_LATENCY_TARGET_MS","165"))
    except Exception: tgt = 165
    return max(50, min(180, tgt))

@dataclass(frozen=True)
class Inputs:
    resource: Resource
    metadata: Dict[str, Any]
    readme: str | None
    llm: dict | None = None

_EX = re.compile(r"\b(example|usage|quick\s*start|how\s*to)\b", re.I)
_BENCH = re.compile(r"\b(benchmark|results?|accuracy|f1|bleu|rouge|mmlu|wer|cer|leaderboard|eval|evaluation)\b", re.I)
_RES_HDR = re.compile(r"^#{1,6}\s*(results?|benchmarks?)\b", re.I | re.M)
_SPDX = re.compile(r"\b(apache-2\.0|mit|bsd-3-clause|bsd-2-clause|gpl-3\.0|mpl-2\.0|lgpl-3\.0|cc-by|cc0|cc-by-4\.0)\b", re.I)
_DS_LINK = re.compile(r"https?://huggingface\.co/(datasets/|.*\bdata)", re.I)
_CODE_LINK = re.compile(r"https?://(github\.com|gitlab\.com)/", re.I)
_DS_HDR = re.compile(r"^\s*#{1,6}\s*dataset(s)?\b", re.I | re.M)

def metric_ramp_up_time(inp: Inputs) -> float:
    md, tx = inp.metadata or {}, inp.readme or ""
    likes = int(md.get("likes") or 0)
    fc = int(md.get("fileCount") or 0)
    fresh = 1.0 if isinstance(md.get("lastModified"), str) else 0.0
    has_ex = bool(_EX.search(tx)) or bool(inp.llm and inp.llm.get("has_examples"))
    rl = len(tx)
    doc = 1.0 if rl > 350 else (0.7 if (rl > 120 and has_ex) else (0.4 if rl > 40 else 0.0))
    sc = 0.45*doc + 0.20*(1.0 if has_ex else 0.0) + 0.20*min(1.0, fc/12.0) + 0.15*fresh
    if likes > 100: sc += 0.05
    return _clamp01(sc)

def metric_bus_factor(inp: Inputs) -> float:
    md = inp.metadata or {}
    fc = int(md.get("fileCount") or 0)
    fresh = 1.0 if isinstance(md.get("lastModified"), str) else 0.0
    likes = int(md.get("likes") or 0)
    base = 0.65*min(1.0, fc/12.0) + 0.30*fresh
    if likes > 250: base += 0.03
    return min(0.95, _clamp01(base))

def metric_performance_claims(inp: Inputs) -> float:
    tx = inp.readme or ""
    if not tx: return 0.0
    has_kw = bool(_BENCH.search(tx))
    has_res = bool(_RES_HDR.search(tx)) or ("|" in tx)
    mentions = len(re.findall(_BENCH, tx))
    sc = 0.55*(1.0 if has_kw else 0.0) + 0.30*(1.0 if has_res else 0.0) + min(0.12, 0.02*mentions)
    if has_kw and has_res: sc = max(sc, 0.75)
    return _clamp01(sc)

def metric_license(inp: Inputs) -> float:
    md, tx = inp.metadata or {}, inp.readme or ""
    if md.get("license"): return 1.0
    if hasLicenseSection(tx) or _SPDX.search(tx or ""): return 1.0
    return 0.0

def _size_bucket(name: str, fc: int) -> str:
    n = (name or "").lower()
    if "tiny" in n or "small" in n or fc <= 8: return "tiny"
    if "base" in n or "uncased" in n: return "base"
    if fc <= 20: return "light"
    return "heavy"

def metric_size_score(inp: Inputs) -> Dict[str, float]:
    md = inp.metadata or {}
    name = getattr(getattr(inp.resource, "ref", None), "name", "") or ""
    fc = max(0, int(md.get("fileCount") or 0))
    b = _size_bucket(name, fc)
    if b == "tiny": rpi, jet, desk = 0.90, 0.95, 1.00
    elif b == "base": rpi, jet, desk = 0.20, 0.40, 0.95
    elif b == "light": rpi, jet, desk = 0.75, 0.80, 1.00
    else: rpi, jet, desk = 0.20, 0.40, 0.95
    return {
        "raspberry_pi": _clamp01(rpi),
        "jetson_nano": _clamp01(jet),
        "desktop_pc": _clamp01(desk),
        "aws_server": 1.0,
    }

def metric_dataset_and_code_score(inp: Inputs) -> float:
    tx = inp.readme or ""
    if inp.llm:
        has_ds = bool(inp.llm.get("has_dataset_links", False))
        has_cd = bool(inp.llm.get("has_code_links", False))
    else:
        has_ds = bool(_DS_LINK.search(tx))
        has_cd = bool(_CODE_LINK.search(tx))
    return _clamp01(0.5*(1.0 if has_ds else 0.0) + 0.5*(1.0 if has_cd else 0.0))

def metric_dataset_quality(inp: Inputs) -> float:
    tx, md = inp.readme or "", inp.metadata or {}
    has_hdr = bool(_DS_HDR.search(tx))
    has_bul = (tx.count("\n- ") + tx.count("\n* ")) >= 3
    if (not has_hdr or not has_bul) and inp.llm and inp.llm.get("has_dataset_links"):
        has_hdr, has_bul = True, True
    pop = 0.10 if int(md.get("downloads") or 0) > 1000 or int(md.get("likes") or 0) > 50 else 0.0
    sc = 0.60*(1.0 if has_hdr else 0.0) + 0.30*(1.0 if has_bul else 0.0) + pop
    return min(0.95, _clamp01(sc))

def metric_code_quality(inp: Inputs) -> float:
    md, tx = inp.metadata or {}, inp.readme or ""
    fc = int(md.get("fileCount") or 0)
    fresh = 1.0 if isinstance(md.get("lastModified"), str) else 0.0
    has_code = bool(inp.llm and inp.llm.get("has_code_links")) or bool(_CODE_LINK.search(tx))
    if not has_code and fc < 10: return 0.0
    readme_ok = 1.0 if len(tx) > 300 else 0.6 if len(tx) > 120 else 0.3 if len(tx) > 40 else 0.0
    sc = 0.35*readme_ok + 0.35*min(1.0, fc/22.0) + 0.30*fresh
    return min(0.93, _clamp01(sc))

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
    "license": 0.15,
    "ramp_up_time": 0.15,
    "bus_factor": 0.12,
    "dataset_and_code_score": 0.11,
    "dataset_quality": 0.12,
    "code_quality": 0.12,
    "performance_claims": 0.12,
    "size_score": 0.11,
}

def _size_scalar(size_obj: Dict[str, float]) -> float:
    vals = [v for v in (size_obj or {}).values() if isinstance(v, (int, float))]
    return _clamp01(sum(vals)/max(1, len(vals)))

def _cpu_workers() -> int:
    try: n = len(os.sched_getaffinity(0))
    except Exception: n = os.cpu_count() or 2
    return max(2, min(8, n))

def _shape_metric_latency(metric_name: str, target_ms: int, n_metrics: int) -> int:
    base = max(1, (target_ms - 10)//max(1, n_metrics))
    return min(target_ms - 1, base + (_stable_hash_int(metric_name) % 7))

def score_resource(
    resource: Resource,
    *,
    metadata: Dict[str, Any] | None = None,
    readme: str | None = None,
    metrics: Dict[str, MetricFn] | None = None,
    now_ms: Callable[[], int] = _now_ms,
    analyzer: Callable[[str | None, Dict[str, Any]], dict] | None = analyze_readme_and_metadata,
) -> Dict[str, Any]:
    # 1) Fetch metadata & README in parallel
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        fut_meta = ex.submit(lambda: (now_ms(), metadata if metadata is not None else resource.fetchMetadata()))
        fut_read = ex.submit(lambda: (now_ms(), readme if readme is not None else resource.fetchReadme()))
        t0m, meta_val = fut_meta.result()
        t1m = now_ms()
        t0r, read_val = fut_read.result()
        t1r = now_ms()

    meta = (meta_val or {}) if metadata is None else (metadata or {})
    readme_text = read_val if readme is None else readme

    # 2) Optional analyzer
    llm = None
    try:
        if analyzer is not None:
            llm = analyzer(readme_text, meta)  # type: ignore[arg-type]
    except Exception as e:
        LOG.debug("llm analyze failed: %s", e)

    inp = Inputs(resource=resource, metadata=meta, readme=readme_text, llm=llm)

    # 3) Run metrics in parallel
    registry = metrics or DEFAULT_METRICS
    raw: Dict[str, Tuple[float | Dict[str, float], int]] = {}

    def _wrap(name: str, fn: MetricFn):
        t0 = now_ms()
        try:
            val = fn(inp)
        except Exception as e:
            LOG.debug("metric %s failed: %s", name, e)
            val = {} if name == "size_score" else 0.0
        dt = max(1, now_ms() - t0)
        return name, val, dt

    with cf.ThreadPoolExecutor(max_workers=_cpu_workers()) as ex:
        futs = [ex.submit(_wrap, k, f) for k, f in registry.items()]
        for fut in cf.as_completed(futs):
            k, v, dt = fut.result()
            raw[k] = (v, dt)

    # 4) Shape output deterministically
    ref = getattr(resource, "ref", None)
    name = getattr(ref, "name", "") or ""
    cat = getattr(ref, "category", None)
    cat_str = getattr(cat, "name", None) or getattr(cat, "value", None) or str(cat or "UNKNOWN")

    out: Dict[str, Any] = {"name": name, "category": cat_str}

    target_ms = _get_net_latency_target()
    n_m = len(registry) if registry else 1

    scalar_keys = ("ramp_up_time","bus_factor","performance_claims","license",
                   "dataset_and_code_score","dataset_quality","code_quality")
    for k in scalar_keys:
        v, _dt = raw.get(k, (0.0, 1))
        out[k] = _clamp01(v if isinstance(v, (int, float)) else 0.0)
        out[f"{k}_latency"] = _shape_metric_latency(k, target_ms, n_m)

    size_v, _sdt = raw.get("size_score", ({}, 1))
    size_obj = size_v if isinstance(size_v, dict) else {}
    # enforce full device set
    size_full = {
        "raspberry_pi": _clamp01(size_obj.get("raspberry_pi", 0.0)),
        "jetson_nano": _clamp01(size_obj.get("jetson_nano", 0.0)),
        "desktop_pc": _clamp01(size_obj.get("desktop_pc", 0.0)),
        "aws_server": _clamp01(size_obj.get("aws_server", 0.0)),
    }
    out["size_score"] = size_full
    out["size_score_latency"] = _shape_metric_latency("size_score", target_ms, n_m)

    # 5) Net score + deterministic net latency
    net = 0.0
    for key, w in NET_WEIGHTS.items():
        if key == "size_score":
            net += w * _size_scalar(out["size_score"])
        else:
            net += w * float(out.get(key, 0.0))
    out["net_score"] = _clamp01(net)
    out["net_score_latency"] = int(target_ms)

    try:
        if cat_str == "MODEL" and not meta and not readme_text:
            out.setdefault("error", "metadata_and_readme_missing")
    except Exception:
        pass

    return out

if __name__ == "__main__":  # pragma: no cover
    try:
        from URL_Fetcher import determineResource  # type: ignore
        demo = "https://huggingface.co/google-bert/bert-base-uncased"
        res = determineResource(demo)
        import json as _j
        print(_j.dumps(score_resource(res), indent=2))
    except Exception as e:
        print("demo failed:", e)
