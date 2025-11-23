"""Lambda handler for POST /artifact/model/{id}/license-check."""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

from src.clients.git_client import GitClient
from src.logging_config import configure_logging
from src.metrics.license import _classify, _load_policy, _normalize_slug
from src.models.artifacts import ArtifactType, validate_artifact_id
from src.storage.errors import ArtifactNotFound
from src.storage.metadata_store import (ArtifactMetadataStore,
                                        build_metadata_store_from_env)
from src.utils.auth import extract_auth_token

configure_logging()
_LOGGER = logging.getLogger(__name__)
_METADATA_STORE: ArtifactMetadataStore = build_metadata_store_from_env()
_GIT_CLIENT: GitClient | None = None


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point for POST /artifact/model/{id}/license-check."""
    try:
        artifact_id = _parse_artifact_id(event)
        _extract_auth_token(event)
        github_url = _parse_body(event)

        artifact = _METADATA_STORE.load(artifact_id)
        if artifact.metadata.type is not ArtifactType.MODEL:
            raise ArtifactNotFound(
                f"Artifact '{artifact_id}' not found for type 'model'"
            )

        compatible = _evaluate_license(github_url)
    except ValueError as error:
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except PermissionError as error:
        return _error_response(HTTPStatus.FORBIDDEN, str(error))
    except ArtifactNotFound as error:
        return _error_response(HTTPStatus.NOT_FOUND, str(error))
    except RuntimeError as error:
        status = _map_external_error(error)
        message = (
            "The artifact or GitHub project could not be found."
            if status is HTTPStatus.NOT_FOUND
            else "External license information could not be retrieved."
        )
        return _error_response(status, message)
    except Exception as error:  # noqa: BLE001 - keep handler resilient
        _LOGGER.exception("Unhandled error in license check handler: %s", error)
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Internal server error",
        )

    return _json_response(HTTPStatus.OK, compatible)


def _parse_artifact_id(event: Dict[str, Any]) -> str:
    artifact_id = (event.get("pathParameters") or {}).get("id")
    if not artifact_id:
        raise ValueError("Path parameter 'id' is required")
    return validate_artifact_id(artifact_id)


def _extract_auth_token(event: Dict[str, Any]) -> str | None:
    return extract_auth_token(event, optional=False)


def _parse_body(event: Dict[str, Any]) -> str:
    body = event.get("body")
    if body is None:
        raise ValueError("Request body is required")
    if event.get("isBase64Encoded"):
        import base64

        try:
            body = base64.b64decode(body).decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("Body could not be decoded from base64") from exc
    if isinstance(body, str):
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON") from exc
    elif isinstance(body, dict):
        payload = body
    else:
        raise ValueError("Request body type is not supported")
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")

    url_value = payload.get("github_url")
    if not isinstance(url_value, str) or not url_value.strip():
        raise ValueError("Field 'github_url' is required and must be a string")
    parsed = urlparse(url_value.strip())
    if not (parsed.scheme and parsed.netloc):
        raise ValueError("Field 'github_url' must be a valid absolute URL")
    return url_value.strip()


def _evaluate_license(github_url: str) -> bool:
    client = _git_client()
    metadata = client.get_repo_metadata(github_url)
    candidates = _extract_license_candidates(metadata)
    if not candidates:
        _LOGGER.info(
            "No license information available for %s; defaulting to incompatible",
            github_url,
        )
        return False
    policy = _load_policy()
    classification = _classify(candidates, policy)
    _LOGGER.info(
        "License classification for %s candidates=%s -> %s",
        github_url,
        candidates,
        classification,
    )
    return classification == "compatible"


def _extract_license_candidates(metadata: Dict[str, Any]) -> List[str]:
    license_field = metadata.get("license")
    values: list[str] = []
    if isinstance(license_field, dict):
        for key in ("spdx_id", "key", "name"):
            value = license_field.get(key)
            if isinstance(value, str):
                values.append(value)
    elif isinstance(license_field, str):
        values.append(license_field)

    normalized = _normalize_candidates(values)
    return normalized


def _normalize_candidates(values: Iterable[str]) -> List[str]:
    normalized: list[str] = []
    for value in values:
        slug = _normalize_slug(value)
        if slug:
            normalized.append(slug)
    # Preserve order but remove duplicates
    seen = set()
    unique: list[str] = []
    for item in normalized:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _map_external_error(error: RuntimeError) -> HTTPStatus:
    msg = str(error).lower()
    if "404" in msg or "not found" in msg:
        return HTTPStatus.NOT_FOUND
    return HTTPStatus.BAD_GATEWAY


def _git_client() -> GitClient:
    global _GIT_CLIENT
    if _GIT_CLIENT is None:
        _GIT_CLIENT = GitClient()
    return _GIT_CLIENT


def _json_response(status: HTTPStatus, body: Any) -> Dict[str, Any]:
    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    return _json_response(status, {"error": message})


# Exposed for tests
def _set_git_client(client: GitClient | None) -> None:
    global _GIT_CLIENT
    _GIT_CLIENT = client
