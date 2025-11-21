"""Lambda handler for POST /artifact/byRegEx searches."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http import HTTPStatus
from typing import Any, Dict, List, Pattern

from src.logging_config import configure_logging
from src.storage.errors import ArtifactNotFound
from src.storage.metadata_store import (ArtifactMetadataStore,
                                        build_metadata_store_from_env)
from src.storage.name_index import (NameIndexEntry, NameIndexStore,
                                    build_name_index_store_from_env)
from src.utils.auth import extract_auth_token
from src.utils.request_logging import log_request

configure_logging()
_LOGGER = logging.getLogger(__name__)
_NAME_INDEX: NameIndexStore = build_name_index_store_from_env()
_METADATA_STORE: ArtifactMetadataStore = build_metadata_store_from_env()
RESULT_LIMIT = 100
PAGE_SIZE = 100
SCAN_TIME_LIMIT = float(os.environ.get("REGEX_SCAN_MAX_SECONDS", "5"))
SCAN_ENTRY_LIMIT = int(os.environ.get("REGEX_SCAN_MAX_ENTRIES", "5000"))
PARALLEL_SEGMENTS = max(
    1, int(os.environ.get("REGEX_SCAN_TOTAL_SEGMENTS", "4"))
)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point for regex-based artifact search."""

    log_request(_LOGGER, event)
    try:
        _extract_auth_token(event)
        regex = _parse_regex(event)
        _LOGGER.info(
            "Regex search start pattern=%s \
             limit=%s", regex.pattern, RESULT_LIMIT
        )
        matches = _search_name_index(regex)
        _LOGGER.info(
            "Regex search complete pattern=%s matches=%s",
            regex.pattern,
            len(matches),
        )
        response = [_artifact_entry_payload(entry) for entry in matches]
    except ValueError as error:
        _log_bad_request(event, error)
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


def _search_name_index(pattern: Pattern[str]) -> List[NameIndexEntry]:
    if _supports_parallel_scan():
        return _parallel_search(pattern)
    results: List[NameIndexEntry] = []
    start_key = None
    scanned_entries = 0
    page_count = 0
    deadline = time.monotonic() + SCAN_TIME_LIMIT
    while len(results) < RESULT_LIMIT:
        entries, start_key = _NAME_INDEX.scan(
            start_key=start_key,
            limit=PAGE_SIZE,
        )
        page_count += 1
        _LOGGER.debug(
            "Regex search page=%s entries=%s next=%s pattern=%s",
            page_count,
            len(entries),
            bool(start_key),
            pattern.pattern,
        )
        for entry in entries:
            scanned_entries += 1
            if _entry_matches(entry, pattern):
                results.append(entry)
                if len(results) >= RESULT_LIMIT:
                    break
            if scanned_entries >= SCAN_ENTRY_LIMIT:
                _LOGGER.info(
                    "Regex search entry cap hit \
                     pattern=%s scanned=%s limit=%s",
                    pattern.pattern,
                    scanned_entries,
                    SCAN_ENTRY_LIMIT,
                )
                return results
        if time.monotonic() >= deadline:
            _LOGGER.info(
                "Regex search time cap hit pattern=%s elapsed=%.2fs",
                pattern.pattern,
                SCAN_TIME_LIMIT,
            )
            break
        if not start_key:
            break
    return results


def _parallel_search(pattern: Pattern[str]) -> List[NameIndexEntry]:
    segments = min(max(PARALLEL_SEGMENTS, 2), 32)
    start_keys: Dict[int, Any | None] = {
        segment: None for segment in range(segments)
    }
    results: List[NameIndexEntry] = []
    scanned_entries = 0
    deadline = time.monotonic() + SCAN_TIME_LIMIT
    page_count = 0

    def _submit_scan(
        executor: ThreadPoolExecutor, segment: int, key: Any
    ) -> Any:
        return executor.submit(
            _NAME_INDEX.scan,
            start_key=key,
            limit=PAGE_SIZE,
            segment=segment,
            total_segments=segments,
        )

    with ThreadPoolExecutor(max_workers=segments) as executor:
        while start_keys and len(results) < RESULT_LIMIT:
            futures = {
                _submit_scan(executor, seg, key): seg
                for seg, key in start_keys.items()
            }
            start_keys = {}
            for future in as_completed(futures):
                segment = futures[future]
                try:
                    entries, next_key = future.result()
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.warning(
                        "Parallel regex scan failed segment=%s error=%s",
                        segment,
                        exc,
                    )
                    continue
                page_count += 1
                _LOGGER.debug(
                    "Regex search page=%s segment=%s entries=%s next=%s pattern=%s",
                    page_count,
                    segment,
                    len(entries),
                    bool(next_key),
                    pattern.pattern,
                )
                for entry in entries:
                    scanned_entries += 1
                    if _entry_matches(entry, pattern):
                        results.append(entry)
                        if len(results) >= RESULT_LIMIT:
                            break
                    if scanned_entries >= SCAN_ENTRY_LIMIT:
                        _LOGGER.info(
                            "Regex search entry cap hit pattern=%s scanned=%s limit=%s",
                            pattern.pattern,
                            scanned_entries,
                            SCAN_ENTRY_LIMIT,
                        )
                        return results
                if next_key is not None and len(results) < RESULT_LIMIT:
                    start_keys[segment] = next_key
                if len(results) >= RESULT_LIMIT:
                    break
            if time.monotonic() >= deadline:
                _LOGGER.info(
                    "Regex search time cap hit pattern=%s elapsed=%.2fs",
                    pattern.pattern,
                    SCAN_TIME_LIMIT,
                )
                break
    return results


def _supports_parallel_scan() -> bool:
    return (
        getattr(_NAME_INDEX, "supports_parallel_scan", False)
        and PARALLEL_SEGMENTS > 1
        and SCAN_TIME_LIMIT > 0
    )


def _entry_matches(entry: NameIndexEntry, pattern: Pattern[str]) -> bool:
    if pattern.search(entry.name):
        return True
    if entry.readme_excerpt and pattern.search(entry.readme_excerpt):
        return True
    return False


def _artifact_entry_payload(entry: NameIndexEntry) -> Dict[str, Any]:
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


def _extract_auth_token(event: Dict[str, Any]) -> str | None:
    return extract_auth_token(event)


def _json_response(status: HTTPStatus, body: Any) -> Dict[str, Any]:
    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    return _json_response(status, {"error": message})


def _log_bad_request(event: Dict[str, Any], error: Exception) -> None:
    body = event.get("body")
    if not isinstance(body, str):
        try:
            body = json.dumps(body)
        except Exception:  # noqa: BLE001
            body = str(body)
    _LOGGER.warning(
        "Bad request for artifact_regex path=%s body=%s error=%s",
        (event.get("requestContext") or {})
        .get("http", {})
        .get("path", event.get("path", "")),
        body,
        error,
    )
