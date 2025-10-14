from __future__ import annotations

"""Helper routines for the top-level `run` launcher."""

import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union

_CLI_MAIN: Optional[Callable[[Optional[Sequence[str]]], int]] = None
PYTHON_BIN = "python3"
PIP_BIN = "pip3"


def _get_cli_main() -> Callable[[Optional[Sequence[str]]], int]:
    """Lazy-load CLI main entry point to avoid import-time dependencies."""
    global _CLI_MAIN
    if _CLI_MAIN is None:
        from .CLIApp import main as cli_main

        _CLI_MAIN = cli_main
    return _CLI_MAIN


def install_dependencies(requirements_path: Path) -> int:
    """Install dependencies declared in `requirements.txt` using pip3."""
    if not requirements_path.exists():
        print(
            f"requirements file not found: {requirements_path}",
            file=sys.stderr,
        )
        return 1

    command = [
        PIP_BIN,
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

    cli_main = _get_cli_main()
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


def _collect_line_coverage(data_path: Optional[Path] = None) -> float:
    """Load the latest coverage results and return the total line coverage."""
    # Local import so ./run install works before dependencies are present.
    from coverage import Coverage

    if data_path is None:
        coverage_api = Coverage()
    else:
        coverage_api = Coverage(data_file=str(data_path))
    try:
        coverage_api.load()
    except FileNotFoundError:
        return 0.0

    buffer = io.StringIO()
    include_patterns = [str(Path.cwd() / "src" / "*")]
    total_percentage = coverage_api.report(
        file=buffer,
        include=include_patterns,
    )
    return total_percentage


Artifact = Union[Path, str]


def _cleanup_coverage_artifacts(
    artifacts: Optional[Iterable[Artifact]] = None,
) -> None:
    """Remove coverage artifacts created during test execution."""
    if os.environ.get("KEEP_COVERAGE"):
        return

    targets: Iterable[Artifact]
    if artifacts is None:
        targets = (Path(".coverage"), Path("coverage.xml"))
    else:
        targets = artifacts

    for artifact in targets:
        path = Path(artifact)
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def _run_pytest_subprocess(
    args: Sequence[str],
    env: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    """Invoke pytest in a subprocess so coverage data is isolated."""

    command = [PYTHON_BIN, "-m", "pytest", *args]
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=dict(env),
        check=False,
    )


def _summarize_pytest_output(stdout: str, stderr: str) -> Tuple[int, int]:
    """Derive passed and total test counts from pytest output text."""

    summary_text = f"{stdout}\n{stderr}"
    pattern = re.compile(
        r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed)"
    )

    counts: dict[str, int] = {}
    for match in pattern.finditer(summary_text):
        count = int(match.group(1))
        key = match.group(2)
        counts[key] = counts.get(key, 0) + count

    passed = counts.get("passed", 0)
    total = sum(counts.values())
    return passed, total


def run_tests() -> int:
    """Execute pytest and print the required summary."""
    coverage_fd, coverage_file = tempfile.mkstemp(
        prefix="coverage-",
        suffix=".dat",
    )
    os.close(coverage_fd)
    coverage_path = Path(coverage_file)
    coverage_xml = coverage_path.with_suffix(".xml")

    pytest_args = [
        "--maxfail=1",
        "--disable-warnings",
        "--quiet",
        "--cov=src",
        "--cov-report=",
    ]
    pytest_env = os.environ.copy()
    pytest_env["COVERAGE_FILE"] = str(coverage_path)

    result = _run_pytest_subprocess(pytest_args, pytest_env)

    if result.returncode != 0:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        _cleanup_coverage_artifacts((coverage_path, coverage_xml))
        return result.returncode

    coverage_percentage = _collect_line_coverage(coverage_path)
    passed, total = _summarize_pytest_output(result.stdout, result.stderr)
    rounded_coverage = round(coverage_percentage)

    summary = (
        f"{passed}/{total} test cases passed. "
        f"{int(rounded_coverage)}% line coverage achieved."
    )
    print(summary)
    _cleanup_coverage_artifacts((coverage_path, coverage_xml))
    return 0


def run_pytest(additional_args: Optional[Sequence[str]] = None) -> int:
    """Run pytest with coverage and stream the full output."""
    command = [
        PYTHON_BIN,
        "-m",
        "pytest",
        "--cov=src",
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point shared between the CLI wrapper and direct invocation."""
    argv = list(argv or sys.argv)
    return dispatch(argv)
