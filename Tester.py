# tester.py
"""
General-purpose test runner with one job:
PRINT EXACTLY: "X/Y test cases passed. Z% line coverage achieved."

Why this file exists:
- Different classes/graders often want a single consistent summary line.
- Teams use pytest, unittest, or both. This runner tries pytest first (nicer),
  and falls back to unittest + coverage if pytest isn't available.

Behavior:
1) Try:    pytest -q --disable-warnings --cov=<pkg> --cov-report=term-missing --cov-report=xml
2) If that fails, try: coverage run -m unittest discover ; coverage report -m ; coverage xml
3) Parse counts (% passed, total collected) and coverage % from stdout.
4) Print the exact summary line and exit with code 0 on success, 1 otherwise.

Notes:
- Keep your package name in PKG under "LIKELY_PACKAGE" or pass it via environment.
- This script is OS-agnostic and should run on Linux or Windows inside your venv.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from typing import Tuple

# If your package is called something else, change this:
LIKELY_PACKAGE = os.getenv("TEST_PKG", "trustcli")


def _run(cmd: str) -> Tuple[int, str]:
    """
    Run a shell command, capture combined stdout/stderr as text.
    Returns (return_code, output_text).
    """
    cp = subprocess.run(
        shlex.split(cmd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return cp.returncode, cp.stdout


def _try_pytest(package: str) -> Tuple[bool, int, int, int, str]:
    """
    Attempt pytest with coverage. Returns:
      (succeeded, passed, total, coverage, raw_output)
    """
    cmd = f"pytest -q --disable-warnings --maxfail=1 --cov={package} --cov-report=term-missing --cov-report=xml"
    rc, out = _run(cmd)

    # Parse "collected N items"
    m_total = re.search(r"collected\s+(\d+)\s+items", out)
    total = int(m_total.group(1)) if m_total else 0

    # Parse "== X passed" (pytest formats vary; this is robust enough for common cases)
    m_passed = re.search(r"==\s*(\d+)\s+passed", out)
    passed = int(m_passed.group(1)) if m_passed else (0 if rc != 0 else total)

    # Parse coverage "TOTAL ... XX%"
    m_cov = re.search(r"TOTAL\s+\d+\s+\d+\s+\d+\s+(\d+)%", out)
    cov = int(m_cov.group(1)) if m_cov else 0

    return (rc == 0), passed, total, cov, out


def _try_unittest() -> Tuple[bool, int, int, int, str]:
    """
    Fallback: unittest + coverage (if installed).
    We run two commands so we can measure coverage:
      1) coverage run -m unittest discover
      2) coverage report -m  (prints a TOTAL line we can parse)
    """
    steps = [
        "coverage run -m unittest discover",
        "coverage report -m",
    ]
    outs = []
    ok = True
    for cmd in steps:
        rc, out = _run(cmd)
        outs.append(out)
        if rc != 0:
            ok = False

    joined = "\n".join(outs)

    # Unittest doesn't print "collected N items". We have to approximate.
    # Try to read lines like "OK" or "Ran N tests in Xs"
    m_ran = re.search(r"Ran\s+(\d+)\s+tests?", joined)
    total = int(m_ran.group(1)) if m_ran else 0
    passed = total if ok else 0

    # Parse coverage TOTAL ... XX%
    m_cov = re.search(r"TOTAL\s+\d+\s+\d+\s+\d+\s+(\d+)%", joined)
    cov = int(m_cov.group(1)) if m_cov else 0

    return ok, passed, total, cov, joined


def main(argv: list[str] | None = None) -> int:
    """
    Entry point. Try pytest first, then unittest.
    Print exactly: "X/Y test cases passed. Z% line coverage achieved."
    Exit 0 on success, 1 on failure.
    """
    pkg = LIKELY_PACKAGE

    ok, passed, total, cov, out = _try_pytest(pkg)
    if not ok:
        # Pytest either failed or doesn't exist -> try unittest fallback
        ok, passed, total, cov, out = _try_unittest()

    # Print the exact single-line summary (the grader may scrape only this)
    print(f"{passed}/{total} test cases passed. {cov}% line coverage achieved.")

    # Exit code mirrors overall success/failure
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
