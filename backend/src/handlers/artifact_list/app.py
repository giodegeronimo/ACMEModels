"""Lambda handler for POST /artifacts listing endpoint."""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Dict, List, Sequence

from src.logging_config import configure_logging
from src.models.artifacts import ArtifactType
from src.storage.errors import ArtifactNotFound
from src.storage.metadata_store import (ArtifactMetadataStore,
                                        build_metadata_store_from_env)
from src.storage.name_index import (NameIndexEntry, NameIndexStore,
                                    build_name_index_store_from_env)
from src.utils.auth import extract_auth_token

configure_logging()
_LOGGER = logging.getLogger(__name__)
_METADATA_STORE: ArtifactMetadataStore = build_metadata_store_from_env()
_NAME_INDEX: NameIndexStore = build_name_index_store_from_env()

MAX_RESULTS = 100
MAX_QUERIES = 25
SCAN_PAGE_SIZE = 100


@dataclass(frozen=True)
class _ParsedQuery:
    name: str
    normalized_name: str
    types: tuple[ArtifactType, ...] | None

    @property
    def is_wildcard(self) -> bool:
        return self.name == "*"


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point for POST /artifacts."""

    try:
        _extract_auth_token(event)
        queries = _parse_queries(event)
        start_key = _parse_offset(event)
        artifacts, next_offset = _collect_matches(queries, start_key)
    except ValueError as error:
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except PermissionError as error:
        return _error_response(HTTPStatus.FORBIDDEN, str(error))
    except Exception as error:  # noqa: BLE001 - keep handler resilient
        _LOGGER.exception(
            "Unhandled error in artifact list handler: %s",
            error,
        )
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Internal server error",
        )

    headers = {"Content-Type": "application/json"}
    if next_offset:
        headers["offset"] = next_offset
    return {
        "statusCode": HTTPStatus.OK.value,
        "headers": headers,
        "body": json.dumps(artifacts),
    }


def _extract_auth_token(event: Dict[str, Any]) -> str | None:
    return extract_auth_token(event)


def _parse_queries(event: Dict[str, Any]) -> List[_ParsedQuery]:
    body = event.get("body")
    if body is None:
        raise ValueError("Request body is required")
    if event.get("isBase64Encoded"):
        body = _decode_base64_body(body)
    if isinstance(body, str):
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON") from exc
    elif isinstance(body, list):
        payload = body
    else:
        raise ValueError("Request body must be a JSON array")

    if not isinstance(payload, list) or not payload:
        raise ValueError("Request body must be a non-empty JSON array")
    if len(payload) > MAX_QUERIES:
        raise ValueError(
            f"Request may include at most {MAX_QUERIES} queries"
        )

    parsed_queries: list[_ParsedQuery] = []
    for item in payload:
        parsed_queries.append(_parse_single_query(item))

    wildcard_queries = [query for query in parsed_queries if query.is_wildcard]
    if wildcard_queries and len(parsed_queries) > 1:
        raise ValueError("Wildcard query '*' must be the only query provided")
    return parsed_queries


def _parse_single_query(raw: Any) -> _ParsedQuery:
    if not isinstance(raw, dict):
        raise ValueError("Each artifact query must be a JSON object")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Each artifact query must include a non-empty name")
    normalized_name = name.casefold()
    types_raw = raw.get("types")
    parsed_types: tuple[ArtifactType, ...] | None = None
    if types_raw is not None:
        if not isinstance(types_raw, list):
            raise ValueError("Field 'types' must be an array when provided")
        if types_raw:
            parsed: list[ArtifactType] = []
            for entry in types_raw:
                if not isinstance(entry, str):
                    raise ValueError("Artifact types must be strings")
                try:
                    parsed.append(ArtifactType(entry))
                except ValueError as exc:
                    raise ValueError(
                        f"Artifact type '{entry}' is invalid"
                    ) from exc
            parsed_types = tuple(sorted(set(parsed), key=lambda t: t.value))
    return _ParsedQuery(
        name=name,
        normalized_name=normalized_name,
        types=parsed_types,
    )


def _decode_base64_body(body: str) -> str:
    try:
        return base64.b64decode(body).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Body could not be decoded from base64") from exc


def _parse_offset(event: Dict[str, Any]) -> Any | None:
    params = event.get("queryStringParameters") or {}
    raw = params.get("offset")
    if not raw:
        return None
    try:
        decoded = base64.urlsafe_b64decode(raw.encode("utf-8")).decode("utf-8")
        payload = json.loads(decoded)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Offset parameter is invalid") from exc
    start_key = payload.get("start_key")
    if start_key is None:
        return None
    return start_key


def _collect_matches(
    queries: Sequence[_ParsedQuery],
    start_key: Any | None,
) -> tuple[list[Dict[str, Any]], str | None]:
    results: list[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    scan_start = start_key

    while True:
        entries, next_start = _NAME_INDEX.scan(
            start_key=scan_start,
            limit=SCAN_PAGE_SIZE,
        )
        if not entries and not next_start:
            break

        for entry in entries:
            if (
                _entry_matches(entry, queries)
                and entry.artifact_id not in seen_ids
            ):
                seen_ids.add(entry.artifact_id)
                results.append(_artifact_payload(entry))
                if len(results) >= MAX_RESULTS:
                    token = _encode_offset_token(_entry_start_key(entry))
                    return results, token
        if not next_start:
            break
        scan_start = next_start

    return results, None


def _entry_matches(
    entry: NameIndexEntry,
    queries: Sequence[_ParsedQuery],
) -> bool:
    for query in queries:
        if query.is_wildcard:
            name_match = True
        else:
            name_match = entry.name.casefold() == query.normalized_name
        if not name_match:
            continue
        if query.types and entry.artifact_type not in query.types:
            continue
        return True
    return False


def _artifact_payload(entry: NameIndexEntry) -> Dict[str, Any]:
    try:
        artifact = _METADATA_STORE.load(entry.artifact_id)
        metadata = artifact.metadata
        return {
            "name": metadata.name,
            "id": metadata.id,
            "type": metadata.type.value,
        }
    except (ArtifactNotFound, Exception):  # noqa: BLE001 - fall back to index
        return {
            "name": entry.name,
            "id": entry.artifact_id,
            "type": entry.artifact_type.value,
        }


def _entry_start_key(entry: NameIndexEntry) -> Dict[str, str]:
    return {
        "normalized_name": entry.normalized_name,
        "artifact_id": entry.artifact_id,
    }


def _encode_offset_token(start_key: Any) -> str:
    payload = json.dumps({"start_key": start_key})
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("utf-8")


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": message}),
    }
