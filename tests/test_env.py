from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.utils import env


def test_load_dotenv_populates_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("GEN_AI_STUDIO_API_KEY=demo-token\n# comment\nEMPTY=\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env, "_ENV_LOADED", False)
    monkeypatch.delenv("GEN_AI_STUDIO_API_KEY", raising=False)

    env.load_dotenv()

    assert os.environ["GEN_AI_STUDIO_API_KEY"] == "demo-token"


def test_load_dotenv_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("GEN_AI_STUDIO_API_KEY=first\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env, "_ENV_LOADED", False)
    monkeypatch.setenv("GEN_AI_STUDIO_API_KEY", "existing")

    env.load_dotenv()
    env.load_dotenv()  # Second call should be a no-op.

    assert os.environ["GEN_AI_STUDIO_API_KEY"] == "existing"


def test_parse_line_helpers() -> None:
    assert env._parse_line("KEY=value") == ("KEY", "value")
    assert env._parse_line("   # comment") is None
    assert env._parse_line("   ") is None
    assert env._parse_line("INVALID") is None


def _reset_env_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "_ENV_LOADED", False)


def test_validate_runtime_environment_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_file = tmp_path / "logs" / "app.log"
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_valid_token")
    monkeypatch.setenv("LOG_FILE", str(log_file))
    _reset_env_module(monkeypatch)

    # Should not raise.
    env.validate_runtime_environment()

    assert log_file.exists()


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("", "GITHUB_TOKEN is empty or unset"),
        ("short", "GITHUB_TOKEN format appears invalid"),
    ],
)
def test_validate_runtime_environment_token_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    token: str,
    expected: str,
) -> None:
    log_file = tmp_path / "app.log"
    monkeypatch.setenv("GITHUB_TOKEN", token)
    monkeypatch.setenv("LOG_FILE", str(log_file))
    _reset_env_module(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        env.validate_runtime_environment()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert expected in captured.err


def test_validate_runtime_environment_logfile_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    existing_dir = tmp_path / "logs"
    log_path = existing_dir / "app.log"
    log_path.mkdir(parents=True)

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token")
    # Point LOG_FILE at a directory with .log suffix so open() fails.
    monkeypatch.setenv("LOG_FILE", str(log_path))
    _reset_env_module(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        env.validate_runtime_environment()

    assert excinfo.value.code == 1


def test_validate_runtime_environment_logfile_extension(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_file = tmp_path / "logs" / "app.txt"
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token")
    monkeypatch.setenv("LOG_FILE", str(log_file))
    _reset_env_module(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        env.validate_runtime_environment()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "LOG_FILE must end with .log" in captured.err
