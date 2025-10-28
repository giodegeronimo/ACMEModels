"""Tests for test rate limiter module."""

from __future__ import annotations

from typing import List

import pytest

from src.net.rate_limiter import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self._now = 0.0
        self.sleeps: List[float] = []

    def time(self) -> float:
        return self._now

    def sleep(self, duration: float) -> None:
        self.sleeps.append(duration)
        self._now += duration


def test_rate_limiter_allows_initial_burst() -> None:
    clock = FakeClock()
    limiter = RateLimiter(
        max_calls=3,
        period_seconds=1.0,
        time_fn=clock.time,
        sleep_fn=clock.sleep,
    )

    # Consume all available tokens without sleeping.
    for _ in range(3):
        limiter.acquire()

    assert clock.sleeps == []


def test_rate_limiter_blocks_when_tokens_exhausted() -> None:
    clock = FakeClock()
    limiter = RateLimiter(
        max_calls=2,
        period_seconds=1.0,
        time_fn=clock.time,
        sleep_fn=clock.sleep,
    )

    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    # Third call should have slept for half a second (time per token).
    assert pytest.approx(clock.sleeps[0], rel=1e-6) == 0.5
    assert pytest.approx(clock.time(), rel=1e-6) == 0.5


def test_rate_limiter_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        RateLimiter(max_calls=0, period_seconds=1.0)

    with pytest.raises(ValueError):
        RateLimiter(max_calls=1, period_seconds=0.0)
