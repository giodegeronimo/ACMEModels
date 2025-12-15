"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Lambda handler for POST /artifact/byRegEx searches.
"""

from __future__ import annotations

import json
import logging
import re
import signal
from http import HTTPStatus
from typing import Any, Callable, Dict, List, Pattern

from src.logging_config import configure_logging
from src.storage.errors import ArtifactNotFound
from src.storage.metadata_store import (ArtifactMetadataStore,
                                        build_metadata_store_from_env)
from src.storage.name_index import (NameIndexEntry, NameIndexStore,
                                    build_name_index_store_from_env)
from src.utils.auth import require_auth_token

configure_logging()
_LOGGER = logging.getLogger(__name__)
_NAME_INDEX: NameIndexStore = build_name_index_store_from_env()
_METADATA_STORE: ArtifactMetadataStore = build_metadata_store_from_env()
RESULT_LIMIT = 100
PAGE_SIZE = 100


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle `POST /artifact/byRegEx` searches.

    The request body includes a `regex` string. We compile it, apply a quick
    timeout guard to reduce catastrophic backtracking risk, then scan the name
    index and return matching artifacts.

    :param event: API Gateway/Lambda proxy event.
    :param context: Lambda context (unused).
    :returns: API Gateway/Lambda proxy response dict.
    """

    try:
        _require_auth(event)
        regex = _parse_regex(event)
        _guard_regex(regex)
        matches = _search_name_index(regex)
        if not matches:
            return _error_response(
                HTTPStatus.NOT_FOUND,
                "No artifact found under this regex",
            )
        response = [_artifact_entry_payload(entry) for entry in matches]
    except ValueError as error:
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except PermissionError as error:
        return _error_response(HTTPStatus.FORBIDDEN, str(error))
    except Exception as error:  # noqa: BLE001 - resilience
        _LOGGER.exception("Unhandled error in regex search handler: %s", error)
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Internal server error",
        )

    return _json_response(HTTPStatus.OK, response)


def _parse_regex(event: Dict[str, Any]) -> Pattern[str]:
    """
    Parse and compile the `regex` string from the request body.

    :param event: API Gateway/Lambda proxy event.
    :returns: Compiled regex pattern.
    :raises ValueError: If the body is missing, malformed, or the regex is invalid.
    """

    body = event.get("body")
    if body is None:
        raise ValueError("Request body is required")
    if isinstance(body, str):
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON") from exc
    elif isinstance(body, dict):
        payload = body
    else:
        raise ValueError("Request body type is not supported")

    regex_value = payload.get("regex")
    if not isinstance(regex_value, str) or not regex_value:
        raise ValueError("Field 'regex' must be a non-empty string")
    try:
        return re.compile(regex_value)
    except re.error as exc:
        raise ValueError(f"Invalid regular expression: {exc}") from exc


def _guard_regex(pattern: Pattern[str]) -> None:
    """
    Validate the regex won't obviously time out on a worst-case-ish probe.

    This is a lightweight mitigation against ReDoS-style patterns. The handler
    also applies timeouts during matching to avoid long-running evaluations.

    :param pattern: Compiled regex pattern to validate.
    :raises ValueError: If evaluation exceeds the timeout.
    """

    def _probe() -> None:
        """
        Run the regex against a probe input likely to trigger backtracking.

        :raises TimeoutError: If the probe evaluation exceeds the timeout.
        """

        test_input = "a" * 5000 + "b"
        pattern.search(test_input)

    try:
        _run_with_timeout(_probe, 0.5)
    except TimeoutError as exc:
        raise ValueError("Regex evaluation exceeded timeout") from exc


def _search_name_index(pattern: Pattern[str]) -> List[NameIndexEntry]:
    """
    Scan the name index and collect entries matching the regex.

    :param pattern: Compiled regex to apply.
    :returns: Matching index entries, capped to `RESULT_LIMIT`.
    """

    results: List[NameIndexEntry] = []
    start_key = None
    while len(results) < RESULT_LIMIT:
        entries, start_key = _NAME_INDEX.scan(
            start_key=start_key,
            limit=PAGE_SIZE,
        )
        for entry in entries:
            if _entry_matches(entry, pattern):
                results.append(entry)
                if len(results) >= RESULT_LIMIT:
                    break
        if not start_key:
            break
    return results


def _entry_matches(entry: NameIndexEntry, pattern: Pattern[str]) -> bool:
    """
    Return True when the entry matches the regex in name or README excerpt.

    Matches are executed under a timeout to reduce the impact of expensive
    regex patterns.

    :param entry: Candidate index entry.
    :param pattern: Compiled regex.
    :returns: True if the entry matches, otherwise False.
    :raises ValueError: If regex evaluation exceeds the timeout.
    """

    def _match() -> bool:
        """
        Perform the actual match logic under the timeout wrapper.

        :returns: Whether the entry matches.
        """

        if pattern.search(entry.name):
            return True
        if entry.readme_excerpt and pattern.search(entry.readme_excerpt):
            return True
        return False

    try:
        return _run_with_timeout(_match, 0.5)
    except TimeoutError as exc:
        raise ValueError("Regex evaluation exceeded timeout") from exc


def _artifact_entry_payload(entry: NameIndexEntry) -> Dict[str, Any]:
    """
    Construct the public response payload for a matching index entry.

    Prefers canonical metadata from the metadata store, but falls back to the
    name index if the metadata store fails (stale entries, transient errors).

    :param entry: Name index entry.
    :returns: JSON-serializable artifact payload.
    """

    try:
        artifact = _METADATA_STORE.load(entry.artifact_id)
        metadata = artifact.metadata
    except (ArtifactNotFound, Exception):  # noqa: BLE001 - fall back to index
        metadata = None
    return {
        "name": metadata.name if metadata else entry.name,
        "id": metadata.id if metadata else entry.artifact_id,
        "type": metadata.type.value
        if metadata
        else entry.artifact_type.value,
    }


def _require_auth(event: Dict[str, Any]) -> None:
    """
    Enforce request authentication for this handler.

    :param event: API Gateway/Lambda proxy event.
    :raises PermissionError: If authorization is missing/invalid.
    """

    require_auth_token(event, optional=False)


def _json_response(status: HTTPStatus, body: Any) -> Dict[str, Any]:
    """
    Create a JSON API Gateway proxy response.

    :param status: HTTP status enum to return.
    :param body: JSON-serializable body.
    :returns: API Gateway/Lambda proxy response dict.
    """

    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _run_with_timeout(func: Callable[[], Any], timeout: float) -> Any:
    """
    Execute a function with a wall-clock timeout using `SIGALRM`.

    Note: `SIGALRM` only works reliably on Unix and in the main thread. This
    is acceptable for the Lambda runtime and unit tests in this repository.

    :param func: Callable to execute.
    :param timeout: Timeout in seconds; non-positive disables the timeout.
    :returns: Return value from `func`.
    :raises TimeoutError: If `func` does not complete in time.
    """

    if timeout <= 0:
        return func()

    def _handler(signum, frame):  # noqa: ANN001
        """
        Signal handler used to interrupt evaluation when the timer fires.
        """

        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        result = func()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    return result


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    """
    Create a JSON error response payload.

    :param status: HTTP status enum to return.
    :param message: Error message to return under the `error` key.
    :returns: API Gateway/Lambda proxy response dict.
    """

    return _json_response(status, {"error": message})
