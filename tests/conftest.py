"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for conftest module.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _enable_readme_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    _enable_readme_fallback: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("ACME_ENABLE_README_FALLBACK", "1")


@pytest.fixture(autouse=True)
def _default_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """
    _default_runtime_env: Function description.
    :param monkeypatch:
    :param tmp_path_factory:
    :returns:
    """

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token_123")
    log_dir = tmp_path_factory.mktemp("logs")
    monkeypatch.setenv("LOG_FILE", str(Path(log_dir) / "acme.log"))
