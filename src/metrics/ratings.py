"""Helpers for computing model ratings via the CLIApp pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

from src.CLIApp import CLIApp

_LOGGER = logging.getLogger(__name__)


class RatingComputationError(RuntimeError):
    """Raised when model ratings cannot be computed."""


def compute_model_rating(hf_url: str) -> Dict[str, Any]:
    """Run CLIApp against ``hf_url`` and return the rating payload."""

    records = [{"hf_url": hf_url}]
    app = CLIApp()

    try:
        results = app.generate_results(records)
    except Exception as exc:  # noqa: BLE001 - propagate context
        _LOGGER.exception(f"Exception occurred while computing rating for '{hf_url}'")
        raise RatingComputationError(
            f"Failed to compute rating for '{hf_url}': {exc}"
        ) from exc

    if not results:
        raise RatingComputationError(
            f"No rating results produced for '{hf_url}'"
        )

    rating = results[0]
    if not isinstance(rating, Mapping):
        raise RatingComputationError(
            f"Rating payload for '{hf_url}' is invalid"
        )

    return dict(rating)
