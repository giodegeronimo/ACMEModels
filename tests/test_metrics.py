from __future__ import annotations

from typing import Dict

import pytest

from src.metrics.base import Metric, MetricOutput
from src.metrics.bus_factor import BusFactorMetric
from src.metrics.code_quality import CodeQualityMetric
from src.metrics.dataset_and_code import DatasetAndCodeMetric
from src.metrics.dataset_quality import DatasetQualityMetric
from src.metrics.license import LicenseMetric
from src.metrics.net_score import NetScoreMetric
from src.metrics.performance import PerformanceMetric
from src.metrics.ramp_up import RampUpMetric
from src.metrics.registry import MetricDispatcher, default_metrics
from src.metrics.size import SizeMetric


@pytest.mark.parametrize(
    "metric_cls",
    [
        NetScoreMetric,
        RampUpMetric,
        BusFactorMetric,
        LicenseMetric,
        DatasetAndCodeMetric,
        DatasetQualityMetric,
        CodeQualityMetric,
        PerformanceMetric,
        SizeMetric,
    ],
)
def test_metric_metadata(metric_cls) -> None:
    metric = metric_cls()
    assert metric.name
    assert metric.key


def test_default_metrics_have_unique_keys() -> None:
    keys = [metric.key for metric in default_metrics()]
    assert len(keys) == len(set(keys))


class StaticMetric(Metric):
    def __init__(self, key: str, value: float) -> None:
        super().__init__(name=f"Static {key}", key=key)
        self._value = value

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        return self._value


class FaultyMetric(Metric):
    def __init__(self) -> None:
        super().__init__(name="Faulty", key="faulty")

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        raise RuntimeError("boom")


def test_dispatcher_collects_results() -> None:
    dispatcher = MetricDispatcher(
        [StaticMetric("a", 0.1), StaticMetric("b", 0.2)]
    )
    results = dispatcher.compute([{}])
    assert len(results) == 1
    keys = [res.key for res in results[0]]
    assert keys == ["a", "b"]
    values = [res.value for res in results[0]]
    assert values == [0.1, 0.2]


def test_dispatcher_handles_errors() -> None:
    dispatcher = MetricDispatcher([FaultyMetric()])
    results = dispatcher.compute([{}])
    faulty = results[0][0]
    assert faulty.key == "faulty"
    assert faulty.value is None
    assert faulty.error == "boom"
    assert faulty.latency_ms >= 0
