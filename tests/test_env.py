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
