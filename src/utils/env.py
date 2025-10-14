from __future__ import annotations

"""Helpers for loading environment configuration."""

import os
from pathlib import Path
from typing import Optional, Tuple

_ENV_LOADED = False


def load_dotenv(dotenv_path: str | Path = ".env") -> None:
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
