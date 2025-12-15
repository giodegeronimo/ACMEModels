"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Lambda handler for GET /artifact/model/{id}/lineage.
"""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from typing import Any, Dict

from src.logging_config import configure_logging
from src.models.artifacts import ArtifactType, validate_artifact_id
from src.models.lineage import ArtifactLineageGraph
from src.storage.errors import ArtifactNotFound, LineageNotFound
from src.storage.lineage_store import load_complete_lineage_family
from src.storage.metadata_store import (ArtifactMetadataStore,
                                        build_metadata_store_from_env)
from src.utils.auth import extract_auth_token

configure_logging()
_LOGGER = logging.getLogger(__name__)
_METADATA_STORE: ArtifactMetadataStore = build_metadata_store_from_env()


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle `GET /artifact/model/{id}/lineage`.

    Loads the artifact record to ensure it exists and is of type `model`, then
    loads the complete lineage family graph from the lineage store.

    :param event: API Gateway/Lambda proxy event.
    :param context: Lambda context (unused).
    :returns: API Gateway/Lambda proxy response dict.
    """

    try:
        artifact_id = _parse_artifact_id(event)
        _extract_auth_token(event)

        artifact = _METADATA_STORE.load(artifact_id)
        if artifact.metadata.type is not ArtifactType.MODEL:
            raise ArtifactNotFound(
                f"Artifact '{artifact_id}' not found for type 'model'"
            )

        lineage_graph = load_complete_lineage_family(artifact_id)
        if lineage_graph is None:
            raise LineageNotFound(
                f"Lineage not found for artifact '{artifact_id}'"
            )
        body = _serialize_lineage_graph(lineage_graph)

    except ValueError as error:
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except PermissionError as error:
        return _error_response(HTTPStatus.FORBIDDEN, str(error))
    except ArtifactNotFound as error:
        return _error_response(HTTPStatus.NOT_FOUND, str(error))
    except LineageNotFound as error:
        return _error_response(HTTPStatus.NOT_FOUND, str(error))
    except Exception as error:  # noqa: BLE001
        _LOGGER.exception("Unhandled error in lineage handler: %s", error)
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Internal server error",
        )

    return _json_response(HTTPStatus.OK, body)


def _parse_artifact_id(event: Dict[str, Any]) -> str:
    """
    Parse and validate the `{id}` path parameter.

    :param event: API Gateway/Lambda proxy event.
    :returns: Validated artifact id string.
    :raises ValueError: If the path parameter is missing or invalid.
    """

    artifact_id = (event.get("pathParameters") or {}).get("id")
    if not artifact_id:
        raise ValueError("Path parameter 'id' is required")
    return validate_artifact_id(artifact_id)


def _extract_auth_token(event: Dict[str, Any]) -> str | None:
    """
    Extract an auth token from the request.

    This uses the shared `extract_auth_token` helper (which may be configured
    to allow optional auth in some environments).

    :param event: API Gateway/Lambda proxy event.
    :returns: Token string when present, otherwise None.
    :raises PermissionError: If auth is required and missing.
    """

    return extract_auth_token(event)


def _serialize_lineage_graph(graph: ArtifactLineageGraph) -> Dict[str, Any]:
    """
    Convert an `ArtifactLineageGraph` into a JSON-serializable dict.

    :param graph: Lineage graph object.
    :returns: Dict with `nodes` and `edges` arrays.
    """

    nodes = []
    for node in graph.nodes:
        node_payload: Dict[str, Any] = {"artifact_id": node.artifact_id}
        if node.name:
            node_payload["name"] = node.name
        if node.source:
            node_payload["source"] = node.source
        if node.metadata:
            node_payload["metadata"] = dict(node.metadata)
        nodes.append(node_payload)

    edges = []
    for edge in graph.edges:
        edges.append(
            {
                "from_node_artifact_id": edge.from_node_artifact_id,
                "to_node_artifact_id": edge.to_node_artifact_id,
                "relationship": edge.relationship,
            }
        )

    return {"nodes": nodes, "edges": edges}


def _json_response(status: HTTPStatus, body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a JSON API Gateway proxy response.

    :param status: HTTP status enum to return.
    :param body: JSON-serializable response body.
    :returns: API Gateway/Lambda proxy response dict.
    """

    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    """
    Create a JSON error response payload.

    :param status: HTTP status enum to return.
    :param message: Error message to return under the `error` key.
    :returns: API Gateway/Lambda proxy response dict.
    """

    return _json_response(status, {"error": message})
