"""Tests for test results module."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, List, Optional

import pytest

from src.metrics.metric_result import MetricResult
from src.results import (OUTPUT_FIELD_ORDER, SIZE_SCORE_DEVICE_ORDER,
                         ResultsFormatter, to_ndjson_line)


def _metric(
    *,
    key: str,
    value: Any,
    latency_ms: int = 12,
    name: Optional[str] = None,
) -> MetricResult:
    metric_name = name or key.replace("_", " ").title()
    return MetricResult(
        metric=metric_name,
        key=key,
        value=value,
        latency_ms=latency_ms,
    )


def test_format_records_skips_entries_without_hf_url() -> None:
    formatter = ResultsFormatter()

    url_records = [
        {"git_url": "https://example.com/git"},
        {"hf_url": "https://huggingface.co/org/model"},
    ]
    metrics: List[List[MetricResult]] = [
        [_metric(key="net_score", value=0.1)],
        [_metric(key="net_score", value=0.9)],
    ]

    formatted = formatter.format_records(url_records, metrics)

    assert len(formatted) == 1
    assert formatted[0]["name"] == "model"


def test_format_records_emits_metric_values_and_latencies() -> None:
    formatter = ResultsFormatter()
    url_records = [
        {
            "hf_url": (
                "https://huggingface.co/google-bert/"
                "bert-base-uncased/tree/main"
            )
        }
    ]
    metrics = [
        [
            _metric(key="net_score", value=0.75, latency_ms=25),
            _metric(
                key="size_score",
                value=MappingProxyType({"aws_server": 1.0}),
                latency_ms=40,
                name="Size Score",
            ),
        ]
    ]

    formatted = formatter.format_records(url_records, metrics)

    record = formatted[0]
    assert record["name"] == "bert-base-uncased"
    assert record["category"] == "MODEL"
    assert record["net_score"] == pytest.approx(0.75)
    assert record["net_score_latency"] == 25

    size_value = record["size_score"]
    assert isinstance(size_value, Dict)
    assert size_value == {"aws_server": 1.0}
    assert record["size_score_latency"] == 40


def test_resolve_name_handles_dataset_prefix() -> None:
    formatter = ResultsFormatter()

    url_records = [
        {"hf_url": "https://huggingface.co/datasets/owner/sample-dataset"}
    ]
    metrics = [[_metric(key="net_score", value=0.2)]]

    formatted = formatter.format_records(url_records, metrics)

    assert formatted[0]["name"] == "sample-dataset"


def test_scores_are_rounded_to_two_decimals() -> None:
    formatter = ResultsFormatter()
    url_records = [{"hf_url": "https://huggingface.co/org/model"}]
    metrics = [
        [
            _metric(key="net_score", value=0.876),
            _metric(
                key="size_score",
                value={"aws_server": 0.995, "jetson_nano": 0.131},
            ),
            _metric(key="bus_factor", value=1),
        ]
    ]

    formatted = formatter.format_records(url_records, metrics)
    record = formatted[0]

    assert record["net_score"] == pytest.approx(0.88)
    size = record["size_score"]
    assert size["aws_server"] == pytest.approx(1.00)
    assert size["jetson_nano"] == pytest.approx(0.13)
    assert record["bus_factor"] == pytest.approx(1.0)


def test_formatter_uses_specified_field_order() -> None:
    formatter = ResultsFormatter()
    url_records = [{"hf_url": "https://huggingface.co/org/model"}]
    metrics = [
        _metric(key="code_quality", value=0.11, latency_ms=70),
        _metric(key="net_score", value=0.2, latency_ms=10),
        _metric(
            key="size_score",
            value={
                "desktop_pc": 0.95,
                "raspberry_pi": 0.2,
                "aws_server": 1.0,
                "jetson_nano": 0.4,
            },
            latency_ms=60,
        ),
        _metric(key="dataset_and_code_score", value=0.3, latency_ms=50),
        _metric(key="bus_factor", value=0.9, latency_ms=30),
        _metric(key="dataset_quality", value=0.6, latency_ms=55),
        _metric(key="license", value=0.7, latency_ms=20),
        _metric(key="performance_claims", value=0.8, latency_ms=40),
        _metric(key="ramp_up_time", value=0.5, latency_ms=25),
    ]

    record = formatter.format_records(url_records, [metrics])[0]

    expected_order = [key for key in OUTPUT_FIELD_ORDER if key in record]
    assert list(record.keys()) == expected_order

    size_keys = list(record["size_score"].keys())
    expected_size_order = [
        key for key in SIZE_SCORE_DEVICE_ORDER if key in record["size_score"]
    ]
    assert size_keys[: len(expected_size_order)] == expected_size_order

    line = to_ndjson_line(record)
    positions = [line.index(f'"{key}"') for key in expected_order]
    assert positions == sorted(positions)


def test_to_ndjson_line_formats_two_decimal_scores() -> None:
    line = to_ndjson_line(
        {
            "name": "model",
            "net_score": 0.8,
            "net_score_latency": 15,
            "size_score": {"aws_server": 1.0},
        }
    )

    assert '"net_score":0.80' in line
    assert '"size_score":{"aws_server":1.00}' in line
    assert '"net_score_latency":15' in line
