"""Base class for rate-limited service clients."""

from __future__ import annotations

import logging
import time
from typing import Callable, Generic, Optional, TypeVar

from src.net.rate_limiter import RateLimiter

T = TypeVar("T")


class BaseClient(Generic[T]):
    """Provide rate-limited execution of outbound requests."""

    def __init__(
        self,
        rate_limiter: RateLimiter,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._rate_limiter = rate_limiter
        self._logger = logger or logging.getLogger(self.__class__.__name__)

    def _execute_with_rate_limit(
        self,
        operation: Callable[[], T],
        *,
        name: Optional[str] = None,
    ) -> T:
        """Run ``operation`` after waiting for rate-limiter availability."""
        label = name or getattr(operation, "__name__", "<anonymous>")
        self._rate_limiter.acquire()

        started_at = time.perf_counter()
        try:
            return operation()
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(
                    "Operation %s completed in %.2f ms",
                    label,
                    elapsed_ms,
                )
