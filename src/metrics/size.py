from __future__ import annotations

from typing import Dict, Mapping

from src.metrics.base import Metric

_DUMMY_SCORE = 0.5


class SizeMetric(Metric):
    """Placeholder size compatibility metric."""

    def __init__(self) -> None:
        super().__init__(name="Size Score", key="size_score")

    def compute(self, url_record: Dict[str, str]) -> Mapping[str, float]:
        return {
            "raspberry_pi": _DUMMY_SCORE,
            "jetson_nano": _DUMMY_SCORE,
            "desktop_pc": _DUMMY_SCORE,
            "aws_server": _DUMMY_SCORE,
        }
