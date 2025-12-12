"""Helpers for computing model ratings via the CLI pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

from src.metrics.net_score import NetScoreCalculator
from src.metrics.registry import MetricDispatcher
from src.results import ResultsFormatter

_LOGGER = logging.getLogger(__name__)


class RatingComputationError(RuntimeError):
    """Raised when model ratings cannot be computed."""


def compute_model_rating(hf_url: str) -> Dict[str, Any]:
    """Run the metric pipeline against ``hf_url`` and return the rating."""

    records = [{"hf_url": hf_url}]
    dispatcher = MetricDispatcher()
    try:
        metric_results = dispatcher.compute(records)
    except Exception as exc:  # noqa: BLE001 - propagate context
        _LOGGER.exception(
            "Exception occurred while computing metrics for '%s'",
            hf_url,
        )
        raise RatingComputationError(
            f"Failed to compute rating for '{hf_url}': {exc}"
        ) from exc

    net_calculator = NetScoreCalculator()
    augmented_results = [
        net_calculator.with_net_score(results)
        for results in metric_results
    ]
    formatter = ResultsFormatter()
    formatted = formatter.format_records(records, augmented_results)

    if not formatted:
        raise RatingComputationError(
            f"No rating results produced for '{hf_url}'"
        )

    rating = formatted[0]
    if not isinstance(rating, Mapping):
        raise RatingComputationError(
            f"Rating payload for '{hf_url}' is invalid"
        )

    return dict(rating)
