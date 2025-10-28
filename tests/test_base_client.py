"""Tests for test base client module."""

from __future__ import annotations

import logging
from typing import Any, List

import pytest

from src.clients.base_client import BaseClient
from src.net.rate_limiter import RateLimiter


class DummyRateLimiter(RateLimiter):
    def __init__(self) -> None:
        # Provide dummy configuration but override acquire.
        super().__init__(max_calls=1, period_seconds=1.0)
        self.calls: List[None] = []

    def acquire(self) -> None:  # type: ignore[override]
        self.calls.append(None)


class SampleClient(BaseClient[Any]):
    def get_value(self) -> int:
        return self._execute_with_rate_limit(lambda: 42, name="get_value")


def test_base_client_executes_operation_and_observes_rate_limit() -> None:
    limiter = DummyRateLimiter()
    client = SampleClient(limiter)

    value = client.get_value()

    assert value == 42
    assert len(limiter.calls) == 1


def test_base_client_logs_latency(caplog: pytest.LogCaptureFixture) -> None:
    limiter = DummyRateLimiter()
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    client = SampleClient(limiter, logger=logger)

    with caplog.at_level(logging.DEBUG, logger="test_logger"):
        client.get_value()

    assert any("get_value" in message for message in caplog.messages)
