from __future__ import annotations

"""Helpers for loading environment configuration."""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

_ENV_LOADED = False


def load_dotenv(dotenv_path: Union[str, Path] = ".env") -> None:
    """Load environment variables from a simple ``.env`` file if present."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    path = Path(dotenv_path)
    if path.exists():
        for line in path.read_text().splitlines():
            parsed = _parse_line(line)
            if parsed:
                key, value = parsed
                os.environ.setdefault(key, value)

    _ENV_LOADED = True


def _parse_line(line: str) -> Optional[Tuple[str, str]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    return (key.strip(), value.strip())


# --- Test and stub controls -------------------------------------------------

def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    lowered = value.strip().lower()
    return lowered in {"1", "true", "yes", "on"}


def ignore_fail_flags() -> bool:
    """Return True when FAIL flags should be ignored (e.g., during tests).

    Controlled by environment variable ``ACME_IGNORE_FAIL``. Any of
    "1/true/yes/on" enables ignoring FAIL.
    """

    load_dotenv()
    return _truthy(os.environ.get("ACME_IGNORE_FAIL"))


def fail_stub_active(flag: bool) -> bool:
    """Return True when a metric's FAIL stub should be used.

    This is a single place to centralize the condition, so tests can disable
    FAIL by setting ``ACME_IGNORE_FAIL=1`` while allowing developers to toggle
    FAIL in code for manual runs.
    """

    return bool(flag and not ignore_fail_flags())
