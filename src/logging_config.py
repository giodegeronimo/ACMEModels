"""Central logging configuration aligned with project requirements."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

_CONFIGURED = False


def configure_logging() -> None:
    """Configure global logging based on LOG_LEVEL and LOG_FILE."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = _read_level(os.getenv("LOG_LEVEL", "0"))
    log_path = os.getenv("LOG_FILE")

    if level is None or level <= 0 or not log_path:
        # Silent mode; keep logging disabled.
        _CONFIGURED = True
        return

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=_map_level(level),
        filename=log_file,
        filemode="a",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _CONFIGURED = True


def _read_level(raw: str) -> Optional[int]:
    try:
        return int(raw)
    except ValueError:
        return None


def _map_level(level: int) -> int:
    if level >= 2:
        return logging.DEBUG
    return logging.INFO
