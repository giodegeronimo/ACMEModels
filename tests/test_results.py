from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, List

import pytest

from src.metrics.metric_result import MetricResult
from src.results import ResultsFormatter


def _metric(
    *,
    key: str,
    value: Any,
    latency_ms: int = 12,
    name: str | None = None,
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
