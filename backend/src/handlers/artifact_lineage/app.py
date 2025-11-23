"""Lambda handler for GET /artifact/model/{id}/lineage."""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from typing import Any, Dict

from src.logging_config import configure_logging
from src.models.artifacts import ArtifactType, validate_artifact_id
from src.models.lineage import ArtifactLineageGraph
from src.storage.errors import ArtifactNotFound, LineageNotFound
from src.storage.lineage_store import (LineageStore,
                                       build_lineage_store_from_env)
from src.storage.metadata_store import (ArtifactMetadataStore,
                                        build_metadata_store_from_env)
from src.utils.auth import extract_auth_token

configure_logging()
_LOGGER = logging.getLogger(__name__)
_METADATA_STORE: ArtifactMetadataStore = build_metadata_store_from_env()
_LINEAGE_STORE: LineageStore = build_lineage_store_from_env()


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point for GET /artifact/model/{id}/lineage."""

    try:
        _log_request(event)
        artifact_id = _parse_artifact_id(event)
        _extract_auth_token(event)
        _LOGGER.info("Parsed lineage request artifact_id=%s", artifact_id)

        # Ensure artifact exists and is of type model
        artifact = _METADATA_STORE.load(artifact_id)
        if artifact.metadata.type is not ArtifactType.MODEL:
            raise ArtifactNotFound(
                f"Artifact '{artifact_id}' not found for type 'model'"
            )
        _LOGGER.debug(
            "Artifact loaded for lineage id=%s name=%s type=%s",
            artifact.metadata.id,
            artifact.metadata.name,
            artifact.metadata.type.value,
        )

        try:
            graph = _LINEAGE_STORE.load(artifact_id)
            _LOGGER.info(
                "Lineage graph loaded artifact_id=%s nodes=%d edges=%d",
                artifact_id,
                len(graph.nodes),
                len(graph.edges),
            )
        except LineageNotFound as exc:
            # Per spec: return 400 when lineage cannot be computed
            raise ValueError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - keep handler resilient
            _LOGGER.exception("Failed to load lineage graph: %s", exc)
            raise ValueError("Lineage metadata malformed") from exc

        body = _serialize_graph(graph)
    except ValueError as error:
        return _error_response(HTTPStatus.BAD_REQUEST, str(error))
    except PermissionError as error:
        return _error_response(HTTPStatus.FORBIDDEN, str(error))
    except ArtifactNotFound as error:
        return _error_response(HTTPStatus.NOT_FOUND, str(error))
    except Exception as error:  # noqa: BLE001
        _LOGGER.exception("Unhandled error in lineage handler: %s", error)
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Internal server error",
        )

    return _json_response(HTTPStatus.OK, body)


def _parse_artifact_id(event: Dict[str, Any]) -> str:
    artifact_id = (event.get("pathParameters") or {}).get("id")
    if not artifact_id:
        raise ValueError("Path parameter 'id' is required")
    return validate_artifact_id(artifact_id)


def _extract_auth_token(event: Dict[str, Any]) -> str | None:
    return extract_auth_token(event)


def _serialize_graph(graph: ArtifactLineageGraph) -> Dict[str, Any]:
    nodes = []
    for n in graph.nodes:
        node: Dict[str, Any] = {"artifact_id": n.artifact_id}
        if n.name is not None:
            node["name"] = n.name
        if n.source is not None:
            node["source"] = n.source
        if n.metadata is not None:
            node["metadata"] = n.metadata
        nodes.append(node)

    edges = []
    for e in graph.edges:
        edges.append(
            {
                "from_node_artifact_id": e.from_node_artifact_id,
                "to_node_artifact_id": e.to_node_artifact_id,
                "relationship": e.relationship,
            }
        )

    return {"nodes": nodes, "edges": edges}


def _log_request(event: Dict[str, Any]) -> None:
    http_ctx = (event.get("requestContext") or {}).get("http", {})
    _LOGGER.info(
        "Lineage request path=%s params=%s headers=%s",
        http_ctx.get("path"),
        event.get("pathParameters"),
        event.get("headers"),
    )


def _json_response(status: HTTPStatus, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status.value,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _error_response(status: HTTPStatus, message: str) -> Dict[str, Any]:
    _LOGGER.info(
        "Lineage request failed status=%s message=%s",
        status.value,
        message,
    )
    return _json_response(status, {"error": message})
