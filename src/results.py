from __future__ import annotations

"""Utilities for transforming metric results into CLI output records."""

from typing import Any, Dict, List, Mapping, Sequence
from urllib.parse import urlparse

from src.metrics.metric_result import MetricResult


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
        output: Dict[str, Any] = {
            "name": self._resolve_model_name(hf_url),
            "category": "MODEL",
        }

        for metric_result in metric_results:
            key = metric_result.key
            value = self._normalize_metric_value(metric_result.value)
            output[key] = value
            output[f"{key}_latency"] = metric_result.latency_ms

        return output

    def _normalize_metric_value(self, value: Any) -> Any:
        if isinstance(value, Mapping) and not isinstance(value, dict):
            return dict(value)
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
