"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Atomic update groups for multi-step artifact operations.

Implements a 3-stage API:
  (1) Begin: initiate an empty transaction group
  (2) Append: add ingest/upload/update steps (optionally with undo callbacks)
  (3) Execute: run all steps under a per-resource lock, rolling back on error

This provides best-effort atomicity for local execution and single-process
deployments. Operations should be ordered so that the system-of-record write
(typically metadata) happens last.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

_T = TypeVar("_T")

_LOCK_DIR = Path(os.environ.get("ACME_LOCK_DIR", "/tmp/acme-locks"))
_LOCK_DIR.mkdir(parents=True, exist_ok=True)


class AtomicUpdateError(RuntimeError):
    """Raised when an atomic update group fails to execute."""


def find_exception_in_chain(
    exc: BaseException,
    types: tuple[type[BaseException], ...],
) -> BaseException | None:
    """
    find_exception_in_chain: Function description.
    :param exc:
    :param types:
    :returns:
    """

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, types):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def _lock_path(key: str) -> Path:
    """
    _lock_path: Function description.
    :param key:
    :returns:
    """

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return _LOCK_DIR / f"{digest}.lock"


@contextlib.contextmanager
def _exclusive_file_lock(path: Path) -> Any:
    """
    _exclusive_file_lock: Function description.
    :param path:
    :returns:
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl  # type: ignore[import-not-found]

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except Exception:
            # Best-effort: fall back to in-process locks only.
            pass
        yield handle
    finally:
        with contextlib.suppress(Exception):
            try:
                import fcntl  # type: ignore[import-not-found]

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        handle.close()


class _LockRegistry:
    """
    _LockRegistry: Class description.
    """

    def __init__(self) -> None:
        """
        __init__: Function description.
        :param:
        :returns:
        """

        self._gate = threading.Lock()
        self._locks: dict[str, threading.RLock] = {}

    @contextlib.contextmanager
    def acquire(self, key: str) -> Any:
        """
        acquire: Function description.
        :param key:
        :returns:
        """

        with self._gate:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._locks[key] = lock
        with lock:
            yield


_LOCKS = _LockRegistry()


@dataclass(frozen=True)
class _Step(Generic[_T]):
    """
    _Step: Class description.
    """

    name: str
    do: Callable[[Dict[str, Any]], _T]
    undo: Optional[Callable[[Dict[str, Any]], None]] = None


class AtomicUpdateGroup:
    """Group multiple operations and execute them with best-effort atomicity."""

    def __init__(self, key: str) -> None:
        """
        __init__: Function description.
        :param key:
        :returns:
        """

        self._key = key
        self._steps: list[_Step[Any]] = []
        self.context: Dict[str, Any] = {}
        self._executed = False

    @classmethod
    def begin(cls, key: str) -> "AtomicUpdateGroup":
        """Stage 1: initiate an empty transaction group."""
        return cls(key)

    def add_step(
        self,
        name: str,
        do: Callable[[Dict[str, Any]], Any],
        *,
        undo: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Stage 2: append an operation to this group."""
        if self._executed:
            raise RuntimeError("Cannot append steps after execute()")
        self._steps.append(_Step(name=name, do=do, undo=undo))

    def execute(self, *, lock_timeout_seconds: float = 30.0) -> Dict[str, Any]:
        """Stage 3: execute all appended operations under an exclusive lock."""
        if self._executed:
            return self.context
        self._executed = True

        deadline = time.monotonic() + max(0.0, lock_timeout_seconds)
        lock_file = _lock_path(self._key)

        while True:
            try:
                with _exclusive_file_lock(lock_file):
                    with _LOCKS.acquire(self._key):
                        return self._execute_steps()
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise AtomicUpdateError(
                        f"Timed out acquiring lock for {self._key}"
                    ) from None
                time.sleep(0.01)

    def _execute_steps(self) -> Dict[str, Any]:
        """
        _execute_steps: Function description.
        :param:
        :returns:
        """

        completed: list[_Step[Any]] = []
        try:
            for step in self._steps:
                step.do(self.context)
                completed.append(step)
        except Exception as exc:  # noqa: BLE001
            for step in reversed(completed):
                if step.undo is None:
                    continue
                with contextlib.suppress(Exception):
                    step.undo(self.context)
            raise AtomicUpdateError(
                f"Atomic update group failed for {self._key}: {exc}"
            ) from exc
        return self.context
