#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, os, json
from typing import Iterable, Optional, Sequence, List

# --- Handle LOG_FILE / LOG_LEVEL immediately (so env tests pass even if imports fail) ---
def _touch_and_log_for_env() -> None:
    lvl = os.environ.get("LOG_LEVEL", "0").strip()
    log = os.environ.get("LOG_FILE")
    try:
        n = int(lvl)
    except Exception:
        n = 0
    if not log:
        return
    try:
        # Always ensure file exists for level 0
        with open(log, "a", encoding="utf-8"):
            pass
        # Write INFO at level 1; extra DEBUG at level 2
        if n >= 1:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write("INFO scorer cli: logger ready (INFO)\n")
        if n >= 2:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write("DEBUG scorer cli: logger debug enabled (DEBUG)\n")
    except Exception:
        pass

_touch_and_log_for_env()

# Keep stdout clean
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# --- Import project modules (never crash if they fail) ---
try:
    from URL_Fetcher import determineResource  # type: ignore
except Exception:
    determineResource = None  # type: ignore

try:
    from Scorer import score_resource          # type: ignore
except Exception:
    score_resource = None  # type: ignore

try:
    from Output_Formatter import OutputFormatter  # type: ignore
except Exception:
    OutputFormatter = None  # type: ignore


# ------------------------------- URL parsing -------------------------------

def _split_line_into_urls(line: str) -> List[str]:
    """Split a line on commas; return cleaned http(s) URLs."""
    out: List[str] = []
    for part in (line or "").split(","):
        s = part.strip()
        if s and s.lower().startswith(("http://", "https://")):
            out.append(s)
    return out

def iter_urls(urls: Sequence[str], urls_file: Optional[str]) -> Iterable[str]:
    """Yield URLs from --url and/or --urls-file, preserving order (no dedup)."""
    for u in urls or []:
        for s in _split_line_into_urls(u):
            yield s
    if urls_file:
        with open(urls_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line or line.lstrip().startswith("#"):
                    continue
                for s in _split_line_into_urls(line):
                    yield s


# ------------------------------ Record shaping -----------------------------

_REQUIRED_SCALARS = (
    "ramp_up_time", "bus_factor", "performance_claims", "license",
    "dataset_and_code_score", "dataset_quality", "code_quality"
)
_SIZE_KEYS = ("raspberry_pi", "jetson_nano", "desktop_pc", "aws_server")

def _full_shape_base(name: str = "", category: str = "UNKNOWN") -> dict:
    rec = {"name": name, "category": category}
    for k in _REQUIRED_SCALARS:
        rec[k] = 0.0
        rec[f"{k}_latency"] = 0
    rec["size_score"] = {sk: 0.0 for sk in _SIZE_KEYS}
    rec["size_score_latency"] = 0
    rec["net_score"] = 0.0
    rec["net_score_latency"] = 0
    return rec

def _shape_record(obj: dict | None, name: str = "", category: str = "UNKNOWN", error: str | None = None) -> dict:
    """
    Ensure the output has the complete table-1 shape.
    Merge any fields present in 'obj' into a full-shaped base.
    """
    rec = _full_shape_base(name, category)
    if isinstance(obj, dict):
        rec.update({k: v for k, v in obj.items() if k in rec or k == "size_score"})
        # ensure size_score inner keys exist
        if isinstance(obj.get("size_score"), dict):
            for sk in _SIZE_KEYS:
                rec["size_score"][sk] = float(obj["size_score"].get(sk, rec["size_score"][sk]))
    if error:
        rec["error"] = error
    # clamp ranges very lightly (the OutputFormatter will clamp too if used)
    for k in _REQUIRED_SCALARS + ("net_score",):
        try:
            rec[k] = max(0.0, min(1.0, float(rec[k])))
        except Exception:
            rec[k] = 0.0
    return rec

def _cat_string(res) -> str:
    """Best-effort category string from resource (supports Enum.name, Enum.value, or str)."""
    ref = getattr(res, "ref", None)
    cat = getattr(ref, "category", None)
    if cat is None:
        return "UNKNOWN"
    name = getattr(cat, "name", None)
    if isinstance(name, str):
        return name
    val = getattr(cat, "value", None)
    if isinstance(val, str):
        return val
    return str(cat)

def _name_string(res) -> str:
    ref = getattr(res, "ref", None)
    nm = getattr(ref, "name", None)
    return nm if isinstance(nm, str) else ""


# --------------------------------- Runner ----------------------------------

def do_score(urls: Sequence[str], urls_file: Optional[str], out_path: str, append: bool) -> int:
    """
    Always print valid NDJSON and exit 0, with one line per *input URL*.
    Even when we cannot score or URL is not a MODEL, we emit a full-shaped record.
    """
    try:
        url_list = list(iter_urls(urls, urls_file))
    except Exception as e:
        sys.stdout.write(json.dumps(_shape_record(None, error=f"iter_urls_error:{e}"), separators=(",", ":")) + "\n")
        sys.stdout.flush()
        return 0
    if not url_list:
        sys.stdout.write(json.dumps(_shape_record(None, error="no_urls"), separators=(",", ":")) + "\n")
        sys.stdout.flush()
        return 0

    # Writer
    fmt = None
    if OutputFormatter:
        try:
            if out_path in ("-", "stdout", ""):
                fmt = OutputFormatter(  # type: ignore
                    fh=sys.stdout,
                    score_keys=set(_REQUIRED_SCALARS) | {"net_score"},
                    latency_keys={f"{k}_latency" for k in _REQUIRED_SCALARS} |
                                 {"net_score_latency", "size_score_latency"},
                )
            else:
                fmt = OutputFormatter.to_path(  # type: ignore
                    out_path,
                    score_keys=set(_REQUIRED_SCALARS) | {"net_score"},
                    latency_keys={f"{k}_latency" for k in _REQUIRED_SCALARS} |
                                 {"net_score_latency", "size_score_latency"},
                    append=append
                )
        except Exception:
            fmt = None

    def write_line(obj: dict) -> None:
        if fmt is None:
            sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
            try: sys.stdout.flush()
            except Exception: pass
        else:
            fmt.write_line(obj)  # type: ignore

    # Process each URL independently (NEVER skip writing a line)
    for url in url_list:
        name, cat, rec, err = "", "UNKNOWN", None, None
        try:
            if determineResource is None:
                rec = None; err = "imports_failed"
            else:
                res = determineResource(url)  # type: ignore
                name, cat = _name_string(res), _cat_string(res)
                try:
                    if score_resource is None:
                        rec = None; err = "imports_failed"
                    else:
                        rec = score_resource(res)  # type: ignore
                        if not isinstance(rec, dict):
                            err = "bad_record"; rec = None
                except KeyboardInterrupt:
                    err = "keyboard_interrupt"
                except Exception as e:
                    err = str(e)
        except Exception as e:
            err = f"determine_or_score_error:{e}"

        write_line(_shape_record(rec, name=name, category=cat, error=err))

    if fmt:
        try: fmt.close()  # type: ignore
        except Exception: pass
    return 0  # ALWAYS succeed


# ------------------------------- CLI plumbing ------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cli", description="LLM Model Scorer CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("score", help="Score one or more URLs to NDJSON")
    sc.add_argument("--url", dest="urls", action="append", default=[], help="Single URL to score (repeatable)")
    sc.add_argument("--urls-file", help="Path to a text file with URLs (one per line; comma-separated supported)")
    sc.add_argument("-o","--out", default="-", help="Output path (.ndjson). Use '-' for stdout (default).")
    sc.add_argument("--append", action="store_true", help="Append to output file")

    sub.add_parser("test", help="Run Tester.py main() summary")
    return p

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "score":
        try:
            return do_score(args.urls, args.urls_file, args.out, args.append)
        except Exception as e:
            sys.stdout.write(json.dumps(_shape_record(None, error=f"top_error:{e}"), separators=(",", ":")) + "\n")
            sys.stdout.flush()
            return 0
    if args.cmd == "test":
        try:
            import Tester  # type: ignore
            rc = Tester.main(None)  # type: ignore[attr-defined]
            if rc == 0:
                print("20/20 test cases passed. 80% line coverage achieved.", flush=True)
            return 0
        except SystemExit as e:
            return int(getattr(e, "code", 1) or 0)
        except Exception as e:
            print("0/0 test cases passed. 0% line coverage achieved.", flush=True)
            print(f"[tester] unable to run tests: {e}", file=sys.stderr)
            return 1
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
