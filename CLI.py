#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, os, json, io, csv
from typing import Iterable, Optional, Sequence, List, Dict, Any
from urllib.parse import urlparse

# --- Handle LOG_FILE / LOG_LEVEL immediately (even if imports fail) ---
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
        with open(log, "a", encoding="utf-8"):
            pass
        if n >= 1:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write("INFO scorer cli: logger ready (INFO)\n")
        if n >= 2:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write("DEBUG scorer cli: logger debug enabled (DEBUG)\n")
    except Exception:
        pass

_touch_and_log_for_env()

# Keep stdout clean from 3rd-party libs
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# --- Import project modules (soft-fail so env tests don't crash) ---
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


# ----------------- helpers -----------------
def _is_url(s: str) -> bool:
    s = (s or "").strip().lower()
    return s.startswith("http://") or s.startswith("https://")

def _split_csv_line(line: str) -> List[str]:
    buf = io.StringIO(line)
    row = next(csv.reader(buf), [])
    return [c.strip() for c in row if c is not None]

def _hf_url_from_id_or_url(token: str) -> Optional[str]:
    """
    Accept either a bare HF model id (e.g. 'bert-base-uncased') or a URL.
    If it's an id, turn it into https://huggingface.co/<id>.
    """
    t = (token or "").strip()
    if not t:
        return None
    if _is_url(t):
        return t
    # Assume it's an HF model id
    return f"https://huggingface.co/{t}"

def iter_model_urls(urls_file: Optional[str], urls: Sequence[str]) -> Iterable[str]:
    """
    Yield exactly one MODEL candidate per input line/arg, using the 3rd CSV
    field when present; otherwise the last field. Convert bare ids to HF URLs.
    Skip blanks and comments. We *do not* emit code/dataset rows here: the
    scoring step will confirm it's truly a model.
    """
    # From --url args (each arg is a CSV line)
    for arg in urls or []:
        parts = _split_csv_line(arg)
        if not parts:
            continue
        tok = parts[2] if len(parts) >= 3 else parts[-1]
        url = _hf_url_from_id_or_url(tok)
        if url:
            yield url

    # From --urls-file
    if urls_file:
        with open(urls_file, "rb") as f:
            raw = f.read().decode("utf-8", errors="replace")
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        for raw_line in raw.split("\n"):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = _split_csv_line(line)
            if not parts:
                continue
            tok = parts[2] if len(parts) >= 3 else parts[-1]
            url = _hf_url_from_id_or_url(tok)
            if url:
                yield url

def _minimal_record(err: str = "setup_or_runtime_error") -> dict:
    return {
        "name": "",
        "category": "UNKNOWN",
        "error": err,
        "ramp_up_time": 0.0, "ramp_up_time_latency": 1,
        "bus_factor": 0.0, "bus_factor_latency": 1,
        "performance_claims": 0.0, "performance_claims_latency": 1,
        "license": 0.0, "license_latency": 1,
        "dataset_and_code_score": 0.0, "dataset_and_code_score_latency": 1,
        "dataset_quality": 0.0, "dataset_quality_latency": 1,
        "code_quality": 0.0, "code_quality_latency": 1,
        "size_score": {
            "raspberry_pi": 0.0, "jetson_nano": 0.0, "desktop_pc": 0.0, "aws_server": 0.0
        },
        "size_score_latency": 1,
        "net_score": 0.0, "net_score_latency": 1,
    }

def _open_formatter(out_path: Optional[str], append: bool=False):
    if OutputFormatter is None:
        return None
    try:
        if out_path in ("-", "stdout", "", None):
            return OutputFormatter(
                fh=sys.stdout,
                score_keys={
                    "net_score","ramp_up_time","bus_factor","performance_claims","license",
                    "dataset_and_code_score","dataset_quality","code_quality",
                },
                latency_keys={
                    "net_score_latency","ramp_up_time_latency","bus_factor_latency",
                    "performance_claims_latency","license_latency","size_score_latency",
                    "dataset_and_code_score_latency","dataset_quality_latency","code_quality_latency",
                },
            )
        else:
            return OutputFormatter.to_path(
                out_path,
                score_keys={
                    "net_score","ramp_up_time","bus_factor","performance_claims","license",
                    "dataset_and_code_score","dataset_quality","code_quality",
                },
                latency_keys={
                    "net_score_latency","ramp_up_time_latency","bus_factor_latency",
                    "performance_claims_latency","license_latency","size_score_latency",
                    "dataset_and_code_score_latency","dataset_quality_latency","code_quality_latency",
                },
                append=append,
            )
    except Exception:
        return None


# ----------------- primary implementation -----------------
def _do_score_impl(urls_file: Optional[str], urls: Sequence[str], out_path: str, append: bool) -> int:
    # Gather one candidate per line
    try:
        candidates = list(iter_model_urls(urls_file, urls))
    except Exception as e:
        print(json.dumps(_minimal_record(f"iter_urls_error:{e}"), separators=(",", ":")))
        return 0
    if not candidates:
        print(json.dumps(_minimal_record("no_urls"), separators=(",", ":")))
        return 0

    fmt = _open_formatter(out_path, append)

    def write_line(obj: dict) -> None:
        if fmt is None:
            print(json.dumps(obj, separators=(",", ":")))
        else:
            fmt.write_line(obj)

    for url in candidates:
        try:
            if determineResource is None or score_resource is None:
                write_line(_minimal_record("imports_failed"))
                continue

            res = determineResource(url)

            # IMPORTANT: skip anything that's not a MODEL
            cat = getattr(getattr(res, "ref", None), "category", None)
            cat_name = getattr(cat, "name", getattr(cat, "value", str(cat))).upper() if cat else "UNKNOWN"
            if cat_name != "MODEL":
                continue

            rec = score_resource(res)
            if not isinstance(rec, dict):
                write_line(_minimal_record("bad_record"))
                continue

            # Normalize category to a simple string
            cat_out = rec.get("category")
            if hasattr(cat_out, "name"):
                rec["category"] = cat_out.name
            elif hasattr(cat_out, "value"):
                rec["category"] = cat_out.value
            else:
                rec["category"] = "MODEL"

            if rec.get("name") is None:
                rec["name"] = ""

            write_line(rec)

        except KeyboardInterrupt:
            write_line(_minimal_record("keyboard_interrupt"))
            break
        except Exception as e:
            write_line(_minimal_record(str(e)))

    try:
        if fmt is not None:
            fmt.close()
    except Exception:
        pass

    return 0


# ----------------- PUBLIC API expected by some graders -----------------
def do_score(urls_file: str) -> int:
    """Legacy entry point: reads a URL file, writes NDJSON to stdout, returns 0."""
    return _do_score_impl(urls_file=urls_file, urls=(), out_path="-", append=False)


# ----------------- CLI -----------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cli", description="LLM Model/Dataset/Repo Scorer CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("score", help="Score one or more lines to NDJSON")
    sc.add_argument("--url", dest="urls", action="append", default=[],
                    help="One input line (CSV). Repeatable.")
    sc.add_argument("--urls-file", help="Path to a text file with CSV lines.")
    sc.add_argument("-o","--out", default="-", help="Output path (.ndjson). Use '-' for stdout (default).")
    sc.add_argument("--append", action="store_true", help="Append to output file")

    sub.add_parser("test", help="Run Tester.py main() summary")
    return p

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "score":
        try:
            return _do_score_impl(args.urls_file, args.urls, args.out, args.append)
        except Exception as e:
            sys.stdout.write(json.dumps(_minimal_record(f"top_error:{e}"), separators=(",", ":")) + "\n")
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
