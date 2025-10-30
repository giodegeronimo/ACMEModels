"""Dispatcher that evaluates metrics for URL records."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence

from src.metrics.base import Metric
from src.metrics.bus_factor import BusFactorMetric
from src.metrics.code_quality import CodeQualityMetric
from src.metrics.dataset_and_code import DatasetAndCodeMetric
from src.metrics.dataset_quality import DatasetQualityMetric
from src.metrics.license import LicenseMetric
from src.metrics.metric_result import MetricResult
from src.metrics.performance import PerformanceMetric
from src.metrics.ramp_up import RampUpMetric
from src.metrics.size import SizeMetric


class MetricDispatcher:
    """Evaluate registered metrics for a sequence of URL records."""

    def __init__(self, metrics: Optional[Iterable[Metric]] = None) -> None:
        if metrics is None:
            metrics = default_metrics()
        self._metrics: List[Metric] = list(metrics)

    @property
    def metrics(self) -> Sequence[Metric]:
        return tuple(self._metrics)

    def compute(
        self, url_records: Sequence[Dict[str, str]]
    ) -> List[List[MetricResult]]:
        return [self._compute_for_record(record) for record in url_records]

    def _compute_for_record(
        self, url_record: Dict[str, str]
    ) -> List[MetricResult]:
        with ThreadPoolExecutor(
            max_workers=len(self._metrics) or 1
        ) as executor:
            futures: List[Future[MetricResult]] = [
                executor.submit(
                    self._execute_metric,
                    metric,
                    url_record,
                )
                for metric in self._metrics
            ]
        return [future.result() for future in futures]

    def _execute_metric(
        self, metric: Metric, url_record: Dict[str, str]
    ) -> MetricResult:
        from time import perf_counter

        start = perf_counter()
        try:
            value = metric.compute(url_record)
            error: Optional[str] = None
        except Exception as exc:  # pragma: no cover - defensive guard
            value = None
            error = str(exc)
        latency_ms = int((perf_counter() - start) * 1000)
        return MetricResult(
            metric=metric.name,
            key=metric.key,
            value=value,
            latency_ms=latency_ms,
            error=error,
        )


def default_metrics() -> List[Metric]:
    return [
        RampUpMetric(),
        BusFactorMetric(),
        LicenseMetric(),
        SizeMetric(),
        DatasetAndCodeMetric(),
        DatasetQualityMetric(),
        CodeQualityMetric(),
        PerformanceMetric(),
    ]
