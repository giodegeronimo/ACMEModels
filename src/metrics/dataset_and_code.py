from __future__ import annotations

from typing import Dict

from src.metrics.base import Metric, MetricOutput

_DUMMY_SCORE = 0.5


class DatasetAndCodeMetric(Metric):
    """Placeholder metric capturing dataset/code availability."""

    def __init__(self) -> None:
        super().__init__(
            name="Dataset & Code Availability",
            key="dataset_and_code_score",
        )

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        return _DUMMY_SCORE
