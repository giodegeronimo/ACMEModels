from __future__ import annotations

"""Helpers for loading environment configuration."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

_ENV_LOADED = False
_README_FALLBACK_DEFAULT = True

_LOGGER = logging.getLogger(__name__)


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


def enable_readme_fallback() -> bool:
    """Return True when README-based fallbacks are permitted."""

    load_dotenv()
    value = os.environ.get("ACME_ENABLE_README_FALLBACK")
    if value is not None:
        return _truthy(value)
    return _README_FALLBACK_DEFAULT


def validate_runtime_environment() -> None:
    """Exit the process when required environment settings are invalid."""

    load_dotenv()

    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    log_path_raw = os.environ.get("LOG_FILE", "").strip()

    def _fail(message: str) -> None:
        _LOGGER.error("Environment validation failed: %s", message)
        print(f"Environment validation failed: {message}", file=sys.stderr)
        raise SystemExit(1)

    if not github_token:
        _fail("GITHUB_TOKEN is empty or unset")

    token_has_expected_shape = (
        len(github_token) >= 8
        and (
            github_token.startswith("ghp_")
            or github_token.startswith("github_")
            or "_" in github_token
        )
    )
    if not token_has_expected_shape:
        _fail("GITHUB_TOKEN format appears invalid")

    if not log_path_raw:
        _fail("LOG_FILE is empty or unset")

    log_path = Path(log_path_raw)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8"):
            pass
    except Exception as error:  # noqa: BLE001 - convert to friendly message
        _fail(f"LOG_FILE is not writable: {error}")

    if not log_path.suffix:
        _LOGGER.warning(
            "LOG_FILE has no extension; continuing but consider using .log",
        )
