"""Lambda handler for GET /artifact/{artifact_type}/{id}/cost."""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from typing import Any, Dict

from src.logging_config import configure_logging
from src.models.artifacts import validate_artifact_id
from src.storage.artifact_cost import (CostCalculationError,
                                       calculate_artifact_cost)
from src.storage.blob_store import BlobNotFoundError
from src.storage.errors import ArtifactNotFound
from src.storage.metadata_store import (ArtifactMetadataStore,
                                        build_metadata_store_from_env)
from src.utils.auth import extract_auth_token

configure_logging()
_LOGGER = logging.getLogger(__name__)
_METADATA_STORE: ArtifactMetadataStore = build_metadata_store_from_env()


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point for GET /artifact/{artifact_type}/{id}/cost."""

    try:
        _log_request(event)
        artifact_id = _parse_artifact_id(event)
        artifact_type = _parse_artifact_type(event)
        include_dependencies = _parse_dependency_param(event)
        _extract_auth_token(event)

        artifact = _METADATA_STORE.load(artifact_id)
        if artifact.metadata.type.value != artifact_type:
            raise ArtifactNotFound(
                f"Artifact '{artifact_id}' not found for type "
                f"'{artifact_type}'"
            )

        lineage_graph = _load_lineage_graph(artifact_id, include_dependencies)
        cost_result = calculate_artifact_cost(
            artifact_id,
            include_dependencies=include_dependencies,
            lineage_graph=lineage_graph,
        )
        _LOGGER.info(
            "Cost calculated artifact=%s include_dependencies=%s result=%s",
            artifact_id,
            include_dependencies,
            cost_result,
        )
        return _json_response(HTTPStatus.OK, cost_result)

    except ValueError as error:
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except PermissionError as error:
        return _error_response(HTTPStatus.FORBIDDEN, str(error))
    except ArtifactNotFound as error:
        return _error_response(HTTPStatus.NOT_FOUND, str(error))
    except BlobNotFoundError as error:
        return _error_response(HTTPStatus.NOT_FOUND, str(error))
    except CostCalculationError as error:
        _LOGGER.exception("Cost calculation error: %s", error)
        return _error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
    except Exception as error:  # noqa: BLE001
        _LOGGER.exception("Unhandled error in cost handler: %s", error)
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Internal server error",
        )


def _parse_artifact_id(event: Dict[str, Any]) -> str:
    artifact_id = (event.get("pathParameters") or {}).get("id")
    if not artifact_id:
        raise ValueError("Path parameter 'id' is required")
    return validate_artifact_id(artifact_id)


def _parse_artifact_type(event: Dict[str, Any]) -> str:
    artifact_type = (event.get("pathParameters") or {}).get("artifact_type")
    if not artifact_type:
        raise ValueError("Path parameter 'artifact_type' is required")
    return artifact_type


def _parse_dependency_param(event: Dict[str, Any]) -> bool:
    """Parse the 'dependency' query parameter (default: false)."""
    query_params = event.get("queryStringParameters") or {}
    dep_value = query_params.get("dependency", "false").lower()
    return dep_value in ("true", "1", "yes")


def _load_lineage_graph(
    artifact_id: str,
    include_dependencies: bool,
) -> Any | None:
    if not include_dependencies:
        return None
    try:
        from src.storage.memory import get_lineage_repo

        lineage_repo = get_lineage_repo()
        return lineage_repo.get(artifact_id)
    except Exception as exc:  # noqa: BLE001 - lineage optional
        _LOGGER.warning(
            "Failed to load lineage for artifact %s: %s",
            artifact_id,
            exc,
        )
        return None


def _extract_auth_token(event: Dict[str, Any]) -> str | None:
    return extract_auth_token(event)


def _log_request(event: Dict[str, Any]) -> None:
    http_ctx = (event.get("requestContext") or {}).get("http", {})
    _LOGGER.info(
        "Cost request path=%s params=%s query=%s headers=%s",
        http_ctx.get("path"),
        event.get("pathParameters"),
        event.get("queryStringParameters"),
        event.get("headers"),
    )


def _json_response(status: HTTPStatus, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    return _json_response(status, {"error": message})
