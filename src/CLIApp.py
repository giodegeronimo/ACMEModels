
"""CLI wiring that parses manifests and emits scored NDJSON records."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .logging_config import configure_logging
from .metrics.net_score import NetScoreCalculator
from .metrics.registry import MetricDispatcher
from .parser import Parser
from .results import ResultsFormatter, to_ndjson_line


class CLIApp:
    """Command-line entry point for the ACME Models tool."""

    def __init__(self, url_file: Optional[Path] = None) -> None:
        self._url_file = Path(url_file) if url_file is not None else None

    def generate_results(
        self,
        url_records: Optional[Sequence[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Compute formatted metric results for the provided URL records."""
        if url_records is None:
            if self._url_file is None:
                raise ValueError(
                    "URL file path is required when records are not provided."
                )
            parser = Parser(self._url_file)
            url_records = parser.parse()

        normalized_records = [dict(record) for record in url_records]

        dispatcher = MetricDispatcher()
        metric_results = dispatcher.compute(normalized_records)

        net_score_calculator = NetScoreCalculator()
        augmented_results = [
            net_score_calculator.with_net_score(results)
            for results in metric_results
        ]

        formatter = ResultsFormatter()
        return formatter.format_records(
            normalized_records,
            augmented_results,
        )

    def run(self) -> int:
        """Execute the CLI workflow and emit NDJSON to stdout."""
        results = self.generate_results()
        for record in results:
            print(to_ndjson_line(record))
        return 0


def _extract_first_value(
    sources: Sequence[Mapping[str, Any]],
    candidate_keys: Sequence[str],
) -> Optional[str]:
    for source in sources:
        for key in candidate_keys:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _normalize_record(source: Mapping[str, Any]) -> Dict[str, str]:
    mapping = {
        "hf_url": ("hf_url", "huggingface_url", "hfUrl", "model_url"),
        "ds_url": ("ds_url", "dataset_url", "datasetUrl"),
        "git_url": ("git_url", "github_url", "gitUrl", "githubUrl"),
    }

    sources: List[Mapping[str, Any]] = [source]
    record: Dict[str, str] = {}
    for target, candidates in mapping.items():
        value = _extract_first_value(sources, candidates)
        if value:
            record[target] = value
    return record


def _extract_records(event: Mapping[str, Any]) -> List[Dict[str, str]]:
    raw_records = event.get("records")
    records: List[Dict[str, str]] = []

    if isinstance(raw_records, Sequence) and not isinstance(
        raw_records,
        (str, bytes),
    ):
        for item in raw_records:
            if isinstance(item, Mapping):
                record = _normalize_record(item)
                if record.get("hf_url"):
                    records.append(record)

    if not records:
        record = _normalize_record(event)
        if record.get("hf_url"):
            records.append(record)

    return records


def _apply_runtime_configuration(event: Mapping[str, Any]) -> None:
    candidates: List[Mapping[str, Any]] = []
    tokens = event.get("tokens")
    if isinstance(tokens, Mapping):
        candidates.append(tokens)
    candidates.append(event)

    env_mapping = {
        "GITHUB_TOKEN": ("github_token", "githubToken"),
        "GEN_AI_STUDIO_API_KEY": (
            "gen_ai_studio_api_key",
            "genAiStudioApiKey",
        ),
        "HUGGINGFACE_HUB_TOKEN": (
            "hf_token",
            "huggingface_token",
            "huggingfaceHubToken",
        ),
        "LOG_LEVEL": ("log_level", "logLevel"),
        "LOG_FILE": ("log_file", "logFile"),
    }

    for env_key, keys in env_mapping.items():
        value = _extract_first_value(candidates, keys)
        if value:
            os.environ[env_key] = value

    os.environ.setdefault("LOG_FILE", "/tmp/acme_models_lambda.log")
    os.environ.setdefault("LOG_LEVEL", "1")


def handler(
    event: Optional[Mapping[str, Any]],
    context: Any,
) -> Dict[str, Any]:
    """AWS Lambda handler that scores model URLs and returns JSON results."""
    del context  # Unused Lambda metadata.
    raw_event = event or {}

    try:
        if not isinstance(raw_event, Mapping):
            raise ValueError("Event payload must be a JSON object.")

        event_payload = dict(raw_event)

        _apply_runtime_configuration(event_payload)
        configure_logging()

        records = _extract_records(event_payload)
        if not records:
            raise ValueError("At least one record with 'hf_url' is required.")

        if not os.environ.get("GITHUB_TOKEN"):
            raise ValueError(
                "github_token is required for GitHub metric access."
            )

        app = CLIApp()
        results = app.generate_results(records)

        response_payload = {"records": results}
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(response_payload),
        }

    except ValueError as error:
        logging.getLogger(__name__).warning("Validation error: %s", error)
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(error)}),
        }
    except Exception as error:  # pragma: no cover - defensive guard
        logging.getLogger(__name__).exception(
            "Unhandled error during Lambda execution."
        )
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(error)}),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description=(
            "ACME Models CLI: Parse URL manifests "
            "into structured data."
        )
    )
    argument_parser.add_argument(
        "url_file",
        type=Path,
        help="Path to the newline-delimited URL manifest file.",
    )
    return argument_parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    argument_parser = build_arg_parser()
    parsed_args = argument_parser.parse_args(argv)

    app = CLIApp(parsed_args.url_file)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
