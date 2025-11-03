
"""CLI wiring that parses manifests and emits scored NDJSON records."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, cast

try:
    import boto3  # type: ignore[import-untyped]
    from botocore.exceptions import (  # type: ignore[import-untyped]
        BotoCoreError, ClientError)
except ImportError:  # pragma: no cover - optional dependency
    boto3 = None  # type: ignore[assignment]
    BotoCoreError = ClientError = Exception

from .logging_config import configure_logging
from .metrics.net_score import NetScoreCalculator
from .metrics.registry import MetricDispatcher
from .parser import Parser
from .results import ResultsFormatter, to_ndjson_line


class CLIApp:
    """Command-line entry point for the ACME Models tool."""

    def __init__(self, url_file: Optional[Path] = None) -> None:
        self._url_file = Path(url_file) if url_file is not None else None
        self._new_results: List[Dict[str, Any]] = []
        self._cached_results: List[Dict[str, Any]] = []

    def generate_results(
        self,
        url_records: Optional[Sequence[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Compute formatted metric results for the provided URL records."""
        self._new_results = []
        self._cached_results = []
        if url_records is None:
            if self._url_file is None:
                raise ValueError(
                    "URL file path is required when records are not provided."
                )
            parser = Parser(self._url_file)
            url_records = parser.parse()

        normalized_records = [dict(record) for record in url_records]

        formatter = ResultsFormatter()
        bucket_name = os.environ.get("MODEL_RESULTS_BUCKET")

        cached_results: Dict[int, Dict[str, Any]] = {}
        records_to_compute: List[Dict[str, str]] = []
        compute_indices: List[int] = []

        if bucket_name and boto3 is not None:
            for index, record in enumerate(normalized_records):
                hf_url = record.get("hf_url")
                if not hf_url:
                    compute_indices.append(index)
                    records_to_compute.append(record)
                    continue
                model_name = formatter._resolve_model_name(hf_url)
                try:
                    cached = _fetch_result_from_s3(model_name, bucket_name)
                except RuntimeError:
                    logger.warning(
                        "Falling back to recomputing metrics for model '%s'.",
                        model_name,
                    )
                    cached = None
                if cached is not None:
                    cached_results[index] = cached
                else:
                    compute_indices.append(index)
                    records_to_compute.append(record)
        else:
            records_to_compute = list(normalized_records)
            compute_indices = list(range(len(normalized_records)))

        dispatcher = MetricDispatcher()
        metric_results: Sequence[Sequence[Any]] = []
        if records_to_compute:
            metric_results = dispatcher.compute(records_to_compute)

        net_score_calculator = NetScoreCalculator()
        augmented_results: Sequence[Sequence[Any]] = []
        if records_to_compute:
            augmented_results = [
                net_score_calculator.with_net_score(results)
                for results in metric_results
            ]

        formatted_results = formatter.format_records(
            records_to_compute if records_to_compute else normalized_records,
            augmented_results if records_to_compute else metric_results,
        )

        final_results: List[Optional[Dict[str, Any]]] = [None] * len(
            normalized_records
        )

        for index, cached in cached_results.items():
            final_results[index] = cached

        if records_to_compute:
            for index, record in zip(compute_indices, formatted_results):
                final_results[index] = record
                self._new_results.append(record)

        if cached_results:
            self._cached_results = [
                cast(Dict[str, Any], final_results[index])
                for index in sorted(cached_results.keys())
                if final_results[index] is not None
            ]

        for index, result in enumerate(final_results):
            if result is None:
                # Should be unreachable but keeps typing strict.
                final_results[index] = formatted_results.pop(0)

        results_list: List[Dict[str, Any]] = []
        for result in final_results:
            if result is not None:
                results_list.append(cast(Dict[str, Any], result))
        return results_list

    def run(self) -> int:
        """Execute the CLI workflow and emit NDJSON to stdout."""
        results = self.generate_results()
        for record in results:
            print(to_ndjson_line(record))
        return 0

    @property
    def new_results(self) -> Sequence[Dict[str, Any]]:
        return list(self._new_results)

    @property
    def cached_results(self) -> Sequence[Dict[str, Any]]:
        return list(self._cached_results)


logger = logging.getLogger(__name__)

DEFAULT_RESULTS_BUCKET = "model-directory-30861"
_S3_CLIENT: Optional[Any] = None


def _sanitize_model_name(model_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "model"


def _get_s3_client() -> Any:
    global _S3_CLIENT
    if boto3 is None:
        raise RuntimeError("boto3 is required to store model results in S3.")
    if _S3_CLIENT is None:
        _S3_CLIENT = boto3.client("s3")
    return _S3_CLIENT


def _store_results_in_s3(
    results: Sequence[Mapping[str, Any]],
    bucket_name: str,
) -> List[str]:
    if not bucket_name:
        raise ValueError(
            "S3 bucket name is required for storing model results."
        )

    client = _get_s3_client()
    stored_keys: List[str] = []

    for record in results:
        model_name = record.get("name")
        if not isinstance(model_name, str) or not model_name.strip():
            logger.warning(
                "Skipping record without valid model name: %s",
                record,
            )
            continue

        object_key = f"{_sanitize_model_name(model_name)}.json"
        try:
            client.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=json.dumps(record).encode("utf-8"),
                ContentType="application/json",
            )
            stored_keys.append(object_key)
        except (BotoCoreError, ClientError) as error:
            logger.exception(
                "Failed to store results for model '%s' in bucket '%s'.",
                model_name,
                bucket_name,
            )
            raise RuntimeError(
                f"Unable to store results for model '{model_name}' in S3."
            ) from error

    return stored_keys


def _fetch_result_from_s3(
    model_name: str,
    bucket_name: str,
) -> Optional[Dict[str, Any]]:
    if not bucket_name:
        return None

    client = _get_s3_client()
    object_key = f"{_sanitize_model_name(model_name)}.json"

    try:
        response = client.get_object(Bucket=bucket_name, Key=object_key)
    except (BotoCoreError, ClientError) as error:
        if isinstance(error, ClientError):
            error_code = error.response.get("Error", {}).get("Code")
            if error_code in {"NoSuchKey", "404"}:
                return None
        logger.exception(
            "Failed to fetch results for model '%s' from bucket '%s'.",
            model_name,
            bucket_name,
        )
        raise RuntimeError(
            f"Unable to fetch results for model '{model_name}' from S3."
        ) from error

    body = response.get("Body")
    if body is None:
        logger.warning(
            "S3 object for model '%s' in bucket '%s' has no body.",
            model_name,
            bucket_name,
        )
        return None

    data = body.read()
    if hasattr(body, "close"):
        body.close()

    try:
        return json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as error:
        logger.exception(
            "Stored JSON for model '%s' in bucket '%s' is invalid.",
            model_name,
            bucket_name,
        )
        raise RuntimeError(
            f"Stored JSON for model '{model_name}' could not be parsed."
        ) from error


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
    os.environ.setdefault("MODEL_RESULTS_BUCKET", DEFAULT_RESULTS_BUCKET)


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

        bucket_name = os.environ.get(
            "MODEL_RESULTS_BUCKET",
            DEFAULT_RESULTS_BUCKET,
        )
        stored_keys: List[str] = []
        if app.new_results:
            stored_keys = _store_results_in_s3(app.new_results, bucket_name)

        response_payload = {
            "records": results,
            "stored_keys": stored_keys,
            "cache_hits": [
                record.get("name") for record in app.cached_results
            ],
        }
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


def get_model_handler(
    event: Optional[Mapping[str, Any]],
    context: Any,
) -> Dict[str, Any]:
    """AWS Lambda handler that returns cached model results from S3."""
    del context
    raw_event = event or {}

    try:
        if not isinstance(raw_event, Mapping):
            raise ValueError("Event payload must be a JSON object.")

        event_payload = dict(raw_event)

        _apply_runtime_configuration(event_payload)
        configure_logging()

        bucket_name = os.environ.get(
            "MODEL_RESULTS_BUCKET",
            DEFAULT_RESULTS_BUCKET,
        )

        model_name = event_payload.get("model_name")
        if isinstance(model_name, str):
            model_name = model_name.strip()

        resolved_name: Optional[str] = model_name if model_name else None
        if not resolved_name:
            records = _extract_records(event_payload)
            if not records:
                raise ValueError("model_name or hf_url is required.")
            record = records[0]
            hf_url = record.get("hf_url")
            if not hf_url:
                raise ValueError("model_name or hf_url is required.")
            formatter = ResultsFormatter()
            resolved_name = formatter._resolve_model_name(hf_url)

        assert resolved_name is not None

        cached = _fetch_result_from_s3(resolved_name, bucket_name)
        if cached is None:
            message = f"No stored results for model '{resolved_name}'."
            return {
                "statusCode": 404,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": message}),
            }

        response_payload = {
            "record": cached,
            "model_name": resolved_name,
            "bucket": bucket_name,
        }
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
    except RuntimeError as error:
        logging.getLogger(__name__).warning("Cache retrieval error: %s", error)
        return {
            "statusCode": 502,
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
