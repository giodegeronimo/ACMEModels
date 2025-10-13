from __future__ import annotations

"""Helper routines for the top-level `run` launcher."""

import io
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .CLIApp import main as cli_main


def install_dependencies(requirements_path: Path) -> int:
    """Install dependencies declared in `requirements.txt` using pip."""
    if not requirements_path.exists():
        print(
            f"requirements file not found: {requirements_path}",
            file=sys.stderr,
        )
        return 1

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements_path),
    ]

    try:
        completed_process = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        return error.returncode

    return completed_process.returncode


def run_parser(url_file: Path) -> int:
    """Run the CLI parser workflow for the provided URL manifest path."""
    if not url_file.exists():
        print(f"URL file not found: {url_file}", file=sys.stderr)
        return 1

    # Mirror the CLI app so ./run stays in sync with direct script usage.
    return cli_main([str(url_file)])


class _PytestStats:
    """Capture pytest statistics for the ./run test summary."""

    def __init__(self) -> None:
        self.passed = 0
        self.total = 0

    def pytest_runtest_logreport(self, report: Any) -> None:
        # Count only the call phase to skip setup/teardown bookkeeping.
        if report.when != "call":
            return

        self.total += 1
        if report.outcome == "passed":
            self.passed += 1


def _collect_line_coverage() -> float:
    """Load the latest coverage results and return the total line coverage."""
    # Local import so ./run install works before dependencies are present.
    from coverage import Coverage

    coverage_api = Coverage()
    try:
        coverage_api.load()
    except FileNotFoundError:
        return 0.0

    buffer = io.StringIO()
    total_percentage = coverage_api.report(file=buffer)
    # Echo the coverage table so developers see the detailed breakdown.
    print(buffer.getvalue().strip())
    return total_percentage


def run_tests() -> int:
    """Execute pytest and print the required summary."""
    try:
        import pytest
    except ModuleNotFoundError as error:  # pragma: no cover - defensive
        print(error, file=sys.stderr)
        return 1

    stats_plugin = _PytestStats()
    pytest_args = [
        "--maxfail=1",
        "--disable-warnings",
        "--cov=.",
        "--cov-report=term",
        "--cov-report=xml",
    ]

    exit_code = pytest.main(pytest_args, plugins=[stats_plugin])
    if exit_code != 0:
        return exit_code

    coverage_percentage = _collect_line_coverage()
    rounded_coverage = round(coverage_percentage)

    summary = (
        f"{stats_plugin.passed}/{stats_plugin.total} test cases passed. "
        f"{int(rounded_coverage)}% line coverage achieved."
    )
    print(summary)
    return 0


def run_pytest(additional_args: Sequence[str] | None = None) -> int:
    """Run pytest with coverage and stream the full output."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=.",
        "--cov-report=term-missing",
    ]
    if additional_args:
        command.extend(additional_args)

    completed = subprocess.run(command)
    return completed.returncode


def dispatch(argv: Sequence[str]) -> int:
    """Route ./run invocations to the appropriate helper."""
    if len(argv) < 2:
        print("Usage: ./run <install|test|pytest|URL_FILE>", file=sys.stderr)
        return 1

    command = argv[1]

    if command == "install":
        requirements_path = Path("requirements.txt")
        return install_dependencies(requirements_path)

    if command == "test":
        return run_tests()

    if command == "pytest":
        return run_pytest(argv[2:])

    url_file = Path(command)
    return run_parser(url_file)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point shared between the CLI wrapper and direct invocation."""
    argv = list(argv or sys.argv)
    return dispatch(argv)
