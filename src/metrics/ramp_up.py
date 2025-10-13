from __future__ import annotations

from typing import Dict

from src.metrics.base import Metric, MetricOutput

_DUMMY_SCORE = 0.5


class RampUpMetric(Metric):
    """Placeholder ramp-up metric."""

    def __init__(self) -> None:
        super().__init__(name="Ramp-up Score", key="ramp_up_time")

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        return _DUMMY_SCORE
