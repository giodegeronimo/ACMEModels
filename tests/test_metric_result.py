"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Coverage for MetricResult helpers.
"""

from __future__ import annotations

from src.metrics.metric_result import MetricResult


def test_metric_result_as_dict_round_trips() -> None:
    """
    test_metric_result_as_dict_round_trips: Function description.
    :param:
    :returns:
    """

    result = MetricResult(
        metric="bus_factor",
        key="bus_factor",
        value=0.7,
        latency_ms=12,
        details={"note": "ok"},
        error=None,
    )

    payload = result.as_dict()

    assert payload["metric"] == "bus_factor"
    assert payload["key"] == "bus_factor"
    assert payload["value"] == 0.7
    assert payload["latency_ms"] == 12
    assert payload["details"] == {"note": "ok"}


def test_metric_result_str_includes_details_and_error() -> None:
    """
    test_metric_result_str_includes_details_and_error: Function description.
    :param:
    :returns:
    """

    result = MetricResult(
        metric="size",
        key="size",
        value=None,
        latency_ms=5,
        details={"count": 3},
        error="boom",
    )

    text = str(result)

    assert "MetricResult" in text
    assert "details" in text
    assert "error" in text
