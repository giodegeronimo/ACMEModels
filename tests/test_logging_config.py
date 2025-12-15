"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Unit tests for logging configuration helper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from src import logging_config


@pytest.fixture(autouse=True)
def _reset_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    _reset_configured: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setattr(logging_config, "_CONFIGURED", False)


def test_read_level_invalid_returns_none() -> None:
    """
    test_read_level_invalid_returns_none: Function description.
    :param:
    :returns:
    """

    assert logging_config._read_level("not-an-int") is None


def test_map_level_thresholds() -> None:
    """
    test_map_level_thresholds: Function description.
    :param:
    :returns:
    """

    assert logging_config._map_level(1) == logging_config.logging.INFO
    assert logging_config._map_level(2) == logging_config.logging.DEBUG


def test_configure_logging_silent_mode_skips_basic_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_configure_logging_silent_mode_skips_basic_config: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("LOG_LEVEL", "0")
    monkeypatch.delenv("LOG_FILE", raising=False)
    calls: list[Dict[str, Any]] = []

    def fake_basic_config(**kwargs: Any) -> None:
        """
        fake_basic_config: Function description.
        :param **kwargs:
        :returns:
        """

        calls.append(kwargs)

    monkeypatch.setattr(logging_config.logging, "basicConfig", fake_basic_config)

    logging_config.configure_logging()

    assert calls == []


def test_configure_logging_configures_stdout_handler_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_configure_logging_configures_stdout_handler_when_enabled: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("LOG_LEVEL", "1")
    monkeypatch.delenv("LOG_FILE", raising=False)
    calls: list[Dict[str, Any]] = []

    def fake_basic_config(**kwargs: Any) -> None:
        """
        fake_basic_config: Function description.
        :param **kwargs:
        :returns:
        """

        calls.append(kwargs)

    monkeypatch.setattr(logging_config.logging, "basicConfig", fake_basic_config)

    logging_config.configure_logging()
    logging_config.configure_logging()

    assert len(calls) == 1
    assert calls[0]["level"] == logging_config.logging.INFO
    assert calls[0]["force"] is True
    assert "filename" not in calls[0]


def test_configure_logging_writes_to_file_when_log_file_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_configure_logging_writes_to_file_when_log_file_set: Function description.
    :param tmp_path:
    :param monkeypatch:
    :returns:
    """

    log_path = tmp_path / "app.log"
    monkeypatch.setenv("LOG_LEVEL", "2")
    monkeypatch.setenv("LOG_FILE", str(log_path))
    calls: list[Dict[str, Any]] = []

    def fake_basic_config(**kwargs: Any) -> None:
        """
        fake_basic_config: Function description.
        :param **kwargs:
        :returns:
        """

        calls.append(kwargs)

    monkeypatch.setattr(logging_config.logging, "basicConfig", fake_basic_config)

    logging_config.configure_logging()

    assert calls and calls[0]["filename"] == log_path
    assert calls[0]["filemode"] == "a"
    assert calls[0]["level"] == logging_config.logging.DEBUG
