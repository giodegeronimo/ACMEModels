"""Lambda handler for POST /artifact/byRegEx searches."""

from __future__ import annotations

import json
import logging
import re
from http import HTTPStatus
from typing import Any, Dict, List, Pattern

from src.storage.errors import ArtifactNotFound
from src.storage.metadata_store import (ArtifactMetadataStore,
                                        build_metadata_store_from_env)
from src.storage.name_index import (NameIndexEntry, NameIndexStore,
                                    build_name_index_store_from_env)

_LOGGER = logging.getLogger(__name__)
_NAME_INDEX: NameIndexStore = build_name_index_store_from_env()
_METADATA_STORE: ArtifactMetadataStore = build_metadata_store_from_env()
RESULT_LIMIT = 100
PAGE_SIZE = 100


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point for regex-based artifact search."""

    try:
        _extract_auth_token(event)
        regex = _parse_regex(event)
        matches = _search_name_index(regex)
        response = [_artifact_entry_payload(entry) for entry in matches]
    except ValueError as error:
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
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
    headers = event.get("headers") or {}
    token = headers.get("X-Authorization") or headers.get("x-authorization")
    if not token:
        _LOGGER.info("Regex search called without X-Authorization header.")
    return token


def _json_response(status: HTTPStatus, body: Any) -> Dict[str, Any]:
    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    return _json_response(status, {"error": message})
