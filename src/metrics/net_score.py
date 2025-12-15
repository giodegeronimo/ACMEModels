"""ACMEModels Repository

Introductory remarks: This module aggregates individual metric results into a
single overall net score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from src.metrics.metric_result import MetricResult

STANDARD_KEY = "net_score"
STANDARD_NAME = "Net Score"


@dataclass(frozen=True)
class NetScoreCalculator:
    """Derive the net score from previously computed metric results."""

    def with_net_score(
        self, metric_results: Iterable[MetricResult]
    ) -> List[MetricResult]:
        """Prepend the computed net score to the provided metric results."""
        results_list = list(metric_results)
        net_score_result = self._create_net_score(results_list)
        return [net_score_result, *results_list]

    def _create_net_score(
        self, metric_results: List[MetricResult]
    ) -> MetricResult:
        """Compute the net score as the average of normalized numeric metrics."""
        numeric_values: List[float] = []
        total_latency = 0

        for metric_result in metric_results:
            total_latency += metric_result.latency_ms
            value = metric_result.value
            if isinstance(value, (int, float)):
                numeric_value = float(value)
                if 0.0 <= numeric_value <= 1.0:
                    numeric_values.append(numeric_value)

        if numeric_values:
            average = sum(numeric_values) / len(numeric_values)
        else:
            average = 0.0

        return MetricResult(
            metric=STANDARD_NAME,
            key=STANDARD_KEY,
            value=round(average, 2),
            latency_ms=total_latency,
        )
