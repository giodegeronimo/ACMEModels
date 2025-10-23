from __future__ import annotations

"""Utilities for transforming metric results into CLI output records."""

import json
from decimal import ROUND_HALF_UP, Decimal
from numbers import Real
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlparse

from src.metrics.metric_result import MetricResult

OUTPUT_FIELD_ORDER: Sequence[str] = (
    "name",
    "category",
    "net_score",
    "net_score_latency",
    "ramp_up_time",
    "ramp_up_time_latency",
    "bus_factor",
    "bus_factor_latency",
    "performance_claims",
    "performance_claims_latency",
    "license",
    "license_latency",
    "size_score",
    "size_score_latency",
    "dataset_and_code_score",
    "dataset_and_code_score_latency",
    "dataset_quality",
    "dataset_quality_latency",
    "code_quality",
    "code_quality_latency",
    "reproducibility",
    "reproducibility_latency",
)

SIZE_SCORE_DEVICE_ORDER: Sequence[str] = (
    "raspberry_pi",
    "jetson_nano",
    "desktop_pc",
    "aws_server",
)


class ResultsFormatter:
    """Format parsed URL records and metric results into NDJSON rows."""

    def format_records(
        self,
        url_records: Sequence[Dict[str, str]],
        metrics_per_record: Sequence[Sequence[MetricResult]],
    ) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for record, metric_results in zip(url_records, metrics_per_record):
            hf_url = record.get("hf_url")
            if not hf_url:
                continue
            formatted.append(self._format_model_record(hf_url, metric_results))
        return formatted

    def _format_model_record(
        self, hf_url: str, metric_results: Sequence[MetricResult]
    ) -> Dict[str, Any]:
        metric_values: Dict[str, Any] = {}
        latency_values: Dict[str, int] = {}
        for metric_result in metric_results:
            key = metric_result.key
            value = self._format_metric_value(key, metric_result.value)
            metric_values[key] = value
            latency_values[f"{key}_latency"] = metric_result.latency_ms

        base_record: Dict[str, Any] = {
            "name": self._resolve_model_name(hf_url),
            "category": "MODEL",
        }
        combined: Dict[str, Any] = {
            **base_record,
            **metric_values,
            **latency_values,
        }

        ordered: Dict[str, Any] = {}
        for field in OUTPUT_FIELD_ORDER:
            if field in combined:
                ordered[field] = combined[field]

        for key, value in combined.items():
            if key not in OUTPUT_FIELD_ORDER:
                ordered[key] = value

        return ordered

    def _format_metric_value(self, key: str, value: Any) -> Any:
        if isinstance(value, Mapping):
            normalized = {
                mapping_key: self._format_score(mapping_value)
                for mapping_key, mapping_value in dict(value).items()
            }
            if key == "size_score":
                ordered_mapping: Dict[str, Any] = {}
                remaining = dict(normalized)
                for device in SIZE_SCORE_DEVICE_ORDER:
                    if device in remaining:
                        ordered_mapping[device] = remaining.pop(device)
                for extra_key in sorted(remaining):
                    ordered_mapping[extra_key] = remaining[extra_key]
                return ordered_mapping
            return normalized
        return self._format_score(value)

    def _format_score(self, value: Any) -> Any:
        if isinstance(value, Real) and not isinstance(value, bool):
            decimal_value = Decimal(str(value))
            rounded = decimal_value.quantize(
                Decimal("0.01"),
                rounding=ROUND_HALF_UP,
            )
            return float(rounded)
        return value

    def _resolve_model_name(self, hf_url: str) -> str:
        parsed = urlparse(hf_url)
        path = parsed.path.strip("/")
        if not path:
            return parsed.netloc or hf_url

        segments = [segment for segment in path.split("/") if segment]
        if not segments:
            return parsed.netloc or hf_url

        if segments[0] == "datasets" and len(segments) > 1:
            segments = segments[1:]

        if "tree" in segments:
            tree_index = segments.index("tree")
            segments = segments[:tree_index]

        if segments:
            return segments[-1]

        return parsed.netloc or hf_url


def to_ndjson_line(record: Mapping[str, Any]) -> str:
    """Serialize a record to JSON with fixed decimal formatting."""

    pairs = [
        f"{_serialize_string(key)}:{_serialize_value(value)}"
        for key, value in record.items()
    ]
    return "{" + ",".join(pairs) + "}"


def _serialize_value(value: Any) -> str:
    if isinstance(value, str):
        return _serialize_string(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, Mapping):
        return "{" + ",".join(
            f"{_serialize_string(str(k))}:{_serialize_value(v)}"
            for k, v in value.items()
        ) + "}"
    if isinstance(value, Iterable) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return "[" + ",".join(_serialize_value(item) for item in value) + "]"
    return str(value)


def _serialize_string(value: str) -> str:
    return json.dumps(value)


def _format_float(value: float) -> str:
    decimal_value = Decimal(str(value))
    rounded = decimal_value.quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )
    return format(rounded, ".2f")
