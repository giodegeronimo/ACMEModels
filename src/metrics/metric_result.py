"""Canonical result object returned by metric computations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

from src.metrics.base import MetricOutput


@dataclass(frozen=True)
class MetricResult:
    metric: str
    key: str
    value: Optional[MetricOutput]
    latency_ms: int
    details: Optional[Mapping[str, Any]] = None
    error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        contents = {
            "metric": self.metric,
            "key": self.key,
            "value": self.value,
            "latency_ms": self.latency_ms,
        }
        if self.details:
            contents["details"] = dict(self.details)
        if self.error:
            contents["error"] = self.error
        return f"MetricResult({contents})"
