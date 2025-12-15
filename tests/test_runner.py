"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for test runner module.
"""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src import runner


class DummyCompletedProcess:
    """
    DummyCompletedProcess: Class description.
    """

    def __init__(self, returncode: int = 0) -> None:
        """
        __init__: Function description.
        :param returncode:
        :returns:
        """

        self.returncode = returncode


def test_install_dependencies_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    test_install_dependencies_missing_file: Function description.
    :param tmp_path:
    :param monkeypatch:
    :param capsys:
    :returns:
    """

    requirements_path = tmp_path / "requirements.txt"
    exit_code = runner.install_dependencies(requirements_path)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "requirements file not found" in captured.err


def test_install_dependencies_invokes_pip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_install_dependencies_invokes_pip: Function description.
    :param tmp_path:
    :param monkeypatch:
    :returns:
    """

    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("pytest\n", encoding="utf-8")

    executed_commands: List[List[str]] = []

    def fake_run(command: List[str], check: bool) -> DummyCompletedProcess:
        """
        fake_run: Function description.
        :param command:
        :param check:
        :returns:
        """

        executed_commands.append(command)
        return DummyCompletedProcess(0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    exit_code = runner.install_dependencies(requirements_path)

    assert exit_code == 0
    expected_prefix = [runner.PIP_BIN, "install", "-r", str(requirements_path)]
    assert executed_commands[0] == expected_prefix


def test_install_dependencies_propagates_pip_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_install_dependencies_propagates_pip_failures: Function description.
    :param tmp_path:
    :param monkeypatch:
    :returns:
    """

    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("pytest\n", encoding="utf-8")

    def fake_run(command: List[str], check: bool) -> DummyCompletedProcess:
        """
        fake_run: Function description.
        :param command:
        :param check:
        :returns:
        """

        raise runner.subprocess.CalledProcessError(returncode=7, cmd=command)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner.install_dependencies(requirements_path) == 7


def test_run_parser_missing_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    test_run_parser_missing_file: Function description.
    :param tmp_path:
    :param capsys:
    :returns:
    """

    url_file = tmp_path / "missing.txt"
    exit_code = runner.run_parser(url_file)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "URL file not found" in captured.err


def test_run_parser_delegates_to_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_run_parser_delegates_to_cli: Function description.
    :param tmp_path:
    :param monkeypatch:
    :returns:
    """

    url_file = tmp_path / "urls.txt"
    url_file.write_text(",,https://hf.co/model\n", encoding="utf-8")

    called_args: List[List[str]] = []

    def fake_cli_main(args: List[str]) -> int:
        """
        fake_cli_main: Function description.
        :param args:
        :returns:
        """

        called_args.append(args)
        return 0

    monkeypatch.setattr(runner, "_CLI_MAIN", fake_cli_main)

    exit_code = runner.run_parser(url_file)

    assert exit_code == 0
    assert called_args == [[str(url_file)]]


def test_dispatch_requires_argument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    test_dispatch_requires_argument: Function description.
    :param capsys:
    :returns:
    """

    exit_code = runner.dispatch(["run"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Usage: ./run" in captured.err


def test_dispatch_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_dispatch_install: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setattr(runner, "install_dependencies", lambda path: 0)
    exit_code = runner.dispatch(["run", "install"])

    assert exit_code == 0


def test_dispatch_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_dispatch_test: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setattr(runner, "run_tests", lambda: 0)
    exit_code = runner.dispatch(["run", "test"])

    assert exit_code == 0


def test_dispatch_pytest(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_dispatch_pytest: Function description.
    :param monkeypatch:
    :returns:
    """

    captured_args: List[List[str]] = []

    def fake_run_pytest(args=None) -> int:
        """
        fake_run_pytest: Function description.
        :param args:
        :returns:
        """

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
    """
    test_dispatch_url_path: Function description.
    :param monkeypatch:
    :param tmp_path:
    :returns:
    """

    url_file = tmp_path / "urls.txt"
    url_file.write_text(",,https://hf.co/model\n", encoding="utf-8")

    monkeypatch.setattr(runner, "run_parser", lambda path: 7)

    exit_code = runner.dispatch(["run", str(url_file)])

    assert exit_code == 7


def test_run_pytest_invokes_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_run_pytest_invokes_subprocess: Function description.
    :param monkeypatch:
    :returns:
    """

    executed_commands: List[List[str]] = []
    injected_env: List[Dict[str, str]] = []

    def fake_run(
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        **_: Any,
    ) -> DummyCompletedProcess:
        """
        fake_run: Function description.
        :param command:
        :param env:
        :param **_:
        :returns:
        """

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
    """
    test_collect_line_coverage_handles_missing_file: Function description.
    :param monkeypatch:
    :returns:
    """

    class FakeCoverage:
        """
        FakeCoverage: Class description.
        """

        def load(self) -> None:
            """
            load: Function description.
            :param:
            :returns:
            """

            raise FileNotFoundError

    fake_module = types.SimpleNamespace(Coverage=FakeCoverage)
    monkeypatch.setitem(sys.modules, "coverage", fake_module)

    assert runner._collect_line_coverage() == 0.0


def test_run_tests_formats_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    test_run_tests_formats_summary: Function description.
    :param monkeypatch:
    :param capsys:
    :returns:
    """

    def fake_collect(_path=None) -> float:
        """
        fake_collect: Function description.
        :param _path:
        :returns:
        """

        return 89.6

    recorded_args: List[List[str]] = []

    class DummyCompletedProcess:
        """
        DummyCompletedProcess: Class description.
        """

        def __init__(self) -> None:
            """
            __init__: Function description.
            :param:
            :returns:
            """

            self.returncode = 0
            self.stdout = "36 passed in 0.10s"
            self.stderr = ""

    def fake_execute(
        args: List[str], env: dict[str, str]
    ) -> DummyCompletedProcess:
        """
        fake_execute: Function description.
        :param args:
        :param env:
        :returns:
        """

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


def test_cleanup_coverage_artifacts_respects_keep_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_cleanup_coverage_artifacts_respects_keep_coverage: Function description.
    :param tmp_path:
    :param monkeypatch:
    :returns:
    """

    target = tmp_path / "coverage.dat"
    target.write_text("data", encoding="utf-8")

    monkeypatch.setenv("KEEP_COVERAGE", "1")
    runner._cleanup_coverage_artifacts([target])
    assert target.exists()

    monkeypatch.delenv("KEEP_COVERAGE", raising=False)
    runner._cleanup_coverage_artifacts([target])
    assert not target.exists()


def test_summarize_pytest_output_counts_failures_and_errors() -> None:
    """
    test_summarize_pytest_output_counts_failures_and_errors: Function description.
    :param:
    :returns:
    """

    passed, total = runner._summarize_pytest_output(
        "2 passed, 1 skipped",
        "1 failed, 3 errors in 0.1s",
    )
    assert passed == 2
    assert total == 7


def test_run_tests_prints_output_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    test_run_tests_prints_output_on_failure: Function description.
    :param monkeypatch:
    :param capsys:
    :returns:
    """

    class DummyCompletedProcess:
        """
        DummyCompletedProcess: Class description.
        """

        def __init__(self) -> None:
            """
            __init__: Function description.
            :param:
            :returns:
            """

            self.returncode = 2
            self.stdout = "some stdout"
            self.stderr = "some stderr"

    monkeypatch.setattr(
        runner,
        "_run_pytest_subprocess",
        lambda args, env: DummyCompletedProcess(),
    )
    cleanup_calls: List[bool] = []
    monkeypatch.setattr(
        runner,
        "_cleanup_coverage_artifacts",
        lambda artifacts=None: cleanup_calls.append(bool(artifacts)),
    )

    assert runner.run_tests() == 2

    captured = capsys.readouterr()
    assert "some stdout" in captured.out
    assert "some stderr" in captured.err
    assert cleanup_calls == [True]


def test_dispatch_web_invalid_port(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    test_dispatch_web_invalid_port: Function description.
    :param monkeypatch:
    :param capsys:
    :returns:
    """

    monkeypatch.setenv("ACME_WEB_PORT", "invalid")
    exit_code = runner.dispatch(["run", "web"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Invalid ACME_WEB_PORT value" in captured.err


def test_dispatch_web_runs_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_dispatch_web_runs_app: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("ACME_WEB_HOST", "0.0.0.0")
    monkeypatch.setenv("ACME_WEB_PORT", "1234")
    monkeypatch.setenv("FLASK_DEBUG", "1")

    calls: List[Dict[str, Any]] = []

    class FakeApp:
        """
        FakeApp: Class description.
        """

        def run(self, *, host: str, port: int, debug: bool) -> None:
            """
            run: Function description.
            :param host:
            :param port:
            :param debug:
            :returns:
            """

            calls.append({"host": host, "port": port, "debug": debug})

    fake_module = types.SimpleNamespace(create_app=lambda: FakeApp())
    monkeypatch.setattr(runner, "_import_from_src", lambda name: fake_module)
    monkeypatch.setattr(runner, "configure_logging", lambda: None)

    assert runner.dispatch(["run", "web"]) == 0
    assert calls == [{"host": "0.0.0.0", "port": 1234, "debug": True}]


def test_runner_module_exec_as_script_adjusts_sys_path() -> None:
    """
    test_runner_module_exec_as_script_adjusts_sys_path: Function description.
    :param:
    :returns:
    """

    runner_path = Path(runner.__file__).resolve()
    original_sys_path = list(sys.path)
    try:
        package_dir = str(Path(runner.__file__).resolve().parent)
        project_root = str(Path(runner.__file__).resolve().parent.parent)
        sys.path[:] = [
            entry
            for entry in sys.path
            if entry not in {package_dir, project_root}
        ]
        globals_dict = runpy.run_path(str(runner_path), run_name="runner_script")
        assert globals_dict["_PKG_PREFIX"] == ""
        assert str(globals_dict["_PACKAGE_DIR"]) in sys.path
        assert str(globals_dict["_PROJECT_ROOT"]) in sys.path
    finally:
        sys.path[:] = original_sys_path


def test_pytest_stats_counts_call_phase_only() -> None:
    """
    test_pytest_stats_counts_call_phase_only: Function description.
    :param:
    :returns:
    """

    stats = runner._PytestStats()

    class Report:
        """
        Report: Class description.
        """

        def __init__(self, when: str, outcome: str) -> None:
            """
            __init__: Function description.
            :param when:
            :param outcome:
            :returns:
            """

            self.when = when
            self.outcome = outcome

    stats.pytest_runtest_logreport(Report("setup", "passed"))
    stats.pytest_runtest_logreport(Report("call", "passed"))
    stats.pytest_runtest_logreport(Report("call", "failed"))
    assert stats.total == 2
    assert stats.passed == 1


def test_collect_line_coverage_reports_when_data_file_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_collect_line_coverage_reports_when_data_file_provided: Function description.
    :param tmp_path:
    :param monkeypatch:
    :returns:
    """

    calls: Dict[str, Any] = {}

    class FakeCoverage:
        """
        FakeCoverage: Class description.
        """

        def __init__(self, data_file: Optional[str] = None) -> None:
            """
            __init__: Function description.
            :param data_file:
            :returns:
            """

            calls["data_file"] = data_file

        def load(self) -> None:
            """
            load: Function description.
            :param:
            :returns:
            """

            return None

        def report(self, *, file, include) -> float:  # type: ignore[no-untyped-def]
            """
            report: Function description.
            :param file:
            :param include:
            :returns:
            """

            calls["include"] = include
            file.write("total\n")
            return 92.3

    monkeypatch.setitem(sys.modules, "coverage", types.SimpleNamespace(Coverage=FakeCoverage))

    data_path = tmp_path / ".coverage"
    assert runner._collect_line_coverage(data_path) == 92.3
    assert calls["data_file"] == str(data_path)
    assert any(str(Path.cwd() / "src") in str(pattern) for pattern in calls["include"])


def test_cleanup_coverage_artifacts_defaults_to_standard_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_cleanup_coverage_artifacts_defaults_to_standard_files: Function description.
    :param tmp_path:
    :param monkeypatch:
    :returns:
    """

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".coverage").write_text("x", encoding="utf-8")
    (tmp_path / "coverage.xml").write_text("x", encoding="utf-8")

    runner._cleanup_coverage_artifacts()

    assert not (tmp_path / ".coverage").exists()
    assert not (tmp_path / "coverage.xml").exists()


def test_run_pytest_subprocess_builds_command_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_run_pytest_subprocess_builds_command_and_env: Function description.
    :param monkeypatch:
    :returns:
    """

    recorded: Dict[str, Any] = {}

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        """
        fake_run: Function description.
        :param command:
        :param **kwargs:
        :returns:
        """

        recorded["command"] = command
        recorded.update(kwargs)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    result = runner._run_pytest_subprocess(["-q"], {"X": "1"})

    assert result.returncode == 0
    assert recorded["command"][:3] == [runner.PYTHON_BIN, "-m", "pytest"]
    assert recorded["command"][-1] == "-q"
    assert recorded["env"]["X"] == "1"
    assert recorded["capture_output"] is True
    assert recorded["text"] is True


def test_main_delegates_to_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_main_delegates_to_dispatch: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setattr(runner, "dispatch", lambda argv: 7)
    assert runner.main(["run", "test"]) == 7


def test_runner_module_main_guard_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_runner_module_main_guard_executes: Function description.
    :param monkeypatch:
    :returns:
    """

    runner_path = Path(runner.__file__).resolve()
    original_argv = list(sys.argv)
    try:
        sys.argv[:] = ["run"]
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(runner_path), run_name="__main__")
        assert exc.value.code == 1
    finally:
        sys.argv[:] = original_argv
