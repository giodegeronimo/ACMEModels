"""Tests for test runner module."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src import runner


class DummyCompletedProcess:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode


def test_install_dependencies_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requirements_path = tmp_path / "requirements.txt"
    exit_code = runner.install_dependencies(requirements_path)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "requirements file not found" in captured.err


def test_install_dependencies_invokes_pip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("pytest\n", encoding="utf-8")

    executed_commands: List[List[str]] = []

    def fake_run(command: List[str], check: bool) -> DummyCompletedProcess:
        executed_commands.append(command)
        return DummyCompletedProcess(0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    exit_code = runner.install_dependencies(requirements_path)

    assert exit_code == 0
    expected_prefix = [runner.PIP_BIN, "install", "-r", str(requirements_path)]
    assert executed_commands[0] == expected_prefix


def test_run_parser_missing_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    url_file = tmp_path / "missing.txt"
    exit_code = runner.run_parser(url_file)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "URL file not found" in captured.err


def test_run_parser_delegates_to_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(",,https://hf.co/model\n", encoding="utf-8")

    called_args: List[List[str]] = []

    def fake_cli_main(args: List[str]) -> int:
        called_args.append(args)
        return 0

    monkeypatch.setattr(runner, "_CLI_MAIN", fake_cli_main)

    exit_code = runner.run_parser(url_file)

    assert exit_code == 0
    assert called_args == [[str(url_file)]]


def test_dispatch_requires_argument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = runner.dispatch(["run"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Usage: ./run" in captured.err


def test_dispatch_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "install_dependencies", lambda path: 0)
    exit_code = runner.dispatch(["run", "install"])

    assert exit_code == 0


def test_dispatch_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "run_tests", lambda: 0)
    exit_code = runner.dispatch(["run", "test"])

    assert exit_code == 0


def test_dispatch_pytest(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_args: List[List[str]] = []

    def fake_run_pytest(args=None) -> int:
        captured_args.append(args)
        return 0

    monkeypatch.setattr(runner, "run_pytest", fake_run_pytest)

    exit_code = runner.dispatch(["run", "pytest", "-q"])

    assert exit_code == 0
    assert captured_args == [["-q"]]


def test_dispatch_url_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(",,https://hf.co/model\n", encoding="utf-8")

    monkeypatch.setattr(runner, "run_parser", lambda path: 7)

    exit_code = runner.dispatch(["run", str(url_file)])

    assert exit_code == 7


def test_run_pytest_invokes_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed_commands: List[List[str]] = []
    injected_env: List[Dict[str, str]] = []

    def fake_run(
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        **_: Any,
    ) -> DummyCompletedProcess:
        executed_commands.append(command)
        if env is not None:
            injected_env.append(env)
        return DummyCompletedProcess(0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    exit_code = runner.run_pytest(["-k", "pattern"])

    assert exit_code == 0
    command = executed_commands[0]
    assert command[:3] == [runner.PYTHON_BIN, "-m", "pytest"]
    assert "--cov=src" in command
    assert command[-2:] == ["-k", "pattern"]
    assert injected_env and injected_env[0].get("ACME_IGNORE_FAIL") == "1"


def test_collect_line_coverage_handles_missing_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCoverage:
        def load(self) -> None:
            raise FileNotFoundError

    fake_module = types.SimpleNamespace(Coverage=FakeCoverage)
    monkeypatch.setitem(sys.modules, "coverage", fake_module)

    assert runner._collect_line_coverage() == 0.0


def test_run_tests_formats_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_collect(_path=None) -> float:
        return 89.6

    recorded_args: List[List[str]] = []

    class DummyCompletedProcess:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = "36 passed in 0.10s"
            self.stderr = ""

    def fake_execute(
        args: List[str], env: dict[str, str]
    ) -> DummyCompletedProcess:
        recorded_args.append(list(args))
        assert "COVERAGE_FILE" in env
        return DummyCompletedProcess()

    monkeypatch.setattr(runner, "_run_pytest_subprocess", fake_execute)
    monkeypatch.setattr(runner, "_collect_line_coverage", fake_collect)
    cleanup_calls: List[bool] = []
    monkeypatch.setattr(
        runner,
        "_cleanup_coverage_artifacts",
        lambda artifacts=None: cleanup_calls.append(bool(artifacts)),
    )

    exit_code = runner.run_tests()
    captured = capsys.readouterr()

    assert exit_code == 0
    expected_line = "36/36 test cases passed. 90% line coverage achieved."
    assert captured.out.strip() == expected_line
    assert captured.err == ""
    assert cleanup_calls == [True]
    assert recorded_args[0][2] == "--quiet"
