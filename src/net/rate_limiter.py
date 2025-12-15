"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Simple token-bucket rate limiter for external service clients.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Optional


class RateLimiter:
    """Enforce a maximum number of operations per time window.

    The limiter implements a token-bucket algorithm. Each call to
    :meth:`acquire` consumes a single token. Tokens refill at a constant rate
    defined by ``max_calls`` over ``period_seconds``.
    """

    def __init__(
        self,
        max_calls: int,
        period_seconds: float,
        *,
        time_fn: Optional[Callable[[], float]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> None:
        """
        __init__: Function description.
        :param max_calls:
        :param period_seconds:
        :param time_fn:
        :param sleep_fn:
        :returns:
        """

        if max_calls <= 0:
            raise ValueError("max_calls must be positive.")
        if period_seconds <= 0:
            raise ValueError("period_seconds must be positive.")

        self._max_calls = float(max_calls)
        self._period_seconds = float(period_seconds)
        self._rate_per_second = self._max_calls / self._period_seconds
        self._time_per_token = self._period_seconds / self._max_calls
        self._time_fn = time_fn or time.monotonic
        self._sleep_fn = sleep_fn or time.sleep

        self._lock = threading.Lock()
        self._tokens = self._max_calls
        self._last_refill = self._time_fn()

    def acquire(self) -> None:
        """Block until a token is available."""
        while True:
            with self._lock:
                now = self._time_fn()
                self._refill_tokens(now)

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                deficit = 1.0 - self._tokens
                wait_time = deficit * self._time_per_token

            # Sleep outside the critical section to avoid blocking
            # other threads.
            self._sleep_fn(wait_time)

    def _refill_tokens(self, now: float) -> None:
        """
        _refill_tokens: Function description.
        :param now:
        :returns:
        """

        elapsed = now - self._last_refill
        if elapsed <= 0:
            return

        replenished = elapsed * self._rate_per_second
        self._tokens = min(self._max_calls, self._tokens + replenished)
        self._last_refill = now
