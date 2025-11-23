"""Persistent storage helpers for lineage graphs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.models.lineage import (ArtifactLineageEdge, ArtifactLineageGraph,
                                ArtifactLineageNode)

from .errors import LineageNotFound, ValidationError

DEFAULT_LINEAGE_DIR = "/tmp/acme-artifact-lineage"


class LineageStoreError(RuntimeError):
    """Raised when lineage persistence fails."""


class LineageStore:
    """Interface for persisting lineage graphs."""

    def save(self, artifact_id: str, graph: ArtifactLineageGraph) -> None:
        raise NotImplementedError

    def load(self, artifact_id: str) -> ArtifactLineageGraph:
        raise NotImplementedError


class LocalLineageStore(LineageStore):
    """File-based lineage store (useful for local/tests)."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, artifact_id: str) -> Path:
        return self._base_dir / f"{artifact_id}.json"

    def save(self, artifact_id: str, graph: ArtifactLineageGraph) -> None:
        path = self._path(artifact_id)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_graph_to_payload(graph), handle)

    def load(self, artifact_id: str) -> ArtifactLineageGraph:
        path = self._path(artifact_id)
        if not path.exists():
            raise LineageNotFound(f"Lineage for '{artifact_id}' not found")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return _payload_to_graph(payload)
        except LineageNotFound:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(
                f"Failed to parse lineage for '{artifact_id}'"
            ) from exc


class S3LineageStore(LineageStore):
    """Persist lineage graphs in S3."""

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "lineage",
        client: Any | None = None,
    ) -> None:
        if not bucket:
            raise ValidationError("bucket name must be provided")
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        if client is None:
            try:
                import boto3 as _boto3
            except ImportError as exc:  # pragma: no cover
                raise LineageStoreError(
                    "boto3 is required for S3 lineage store"
                ) from exc
            region = (
                os.environ.get("ARTIFACT_STORAGE_REGION")
                or os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
            )
            client_kwargs: dict[str, Any] = {}
            if region:
                client_kwargs["region_name"] = region
            client = _boto3.client("s3", **client_kwargs)
        self._s3 = client

    def _key(self, artifact_id: str) -> str:
        prefix = f"{self._prefix}/" if self._prefix else ""
        return f"{prefix}{artifact_id}.json"

    def save(self, artifact_id: str, graph: ArtifactLineageGraph) -> None:
        key = self._key(artifact_id)
        try:
            payload = json.dumps(_graph_to_payload(graph)).encode("utf-8")
            self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=payload,
                ContentType="application/json",
            )
        except Exception as exc:  # noqa: BLE001
            raise LineageStoreError(
                f"Failed to write lineage: {exc}"
            ) from exc

    def load(self, artifact_id: str) -> ArtifactLineageGraph:
        key = self._key(artifact_id)
        try:
            response = self._s3.get_object(Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            if _is_access_denied(exc) or _is_not_found(exc):
                raise LineageNotFound(
                    f"Lineage for '{artifact_id}' not found"
                ) from exc
            raise LineageStoreError(
                f"Failed to read lineage: {exc}"
            ) from exc
        payload = json.loads(response["Body"].read())
        try:
            return _payload_to_graph(payload)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(
                f"Failed to parse lineage for '{artifact_id}'"
            ) from exc


def _is_access_denied(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    if not isinstance(error, dict):
        return False
    code = error.get("Code")
    if not isinstance(code, str):
        return False
    return code == "AccessDenied"


def _is_not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    if not isinstance(error, dict):
        return False
    code = error.get("Code")
    if not isinstance(code, str):
        return False
    return code in {"404", "NoSuchKey", "NotFound"}


def _graph_to_payload(graph: ArtifactLineageGraph) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "artifact_id": node.artifact_id,
                **({"name": node.name} if node.name is not None else {}),
                **(
                    {"source": node.source} if node.source is not None else {}
                ),
                **(
                    {"metadata": node.metadata}
                    if node.metadata is not None
                    else {}
                ),
            }
            for node in graph.nodes
        ],
        "edges": [
            {
                "from_node_artifact_id": edge.from_node_artifact_id,
                "to_node_artifact_id": edge.to_node_artifact_id,
                "relationship": edge.relationship,
            }
            for edge in graph.edges
        ],
    }


def _payload_to_graph(payload: dict[str, Any]) -> ArtifactLineageGraph:
    nodes_raw = payload.get("nodes")
    edges_raw = payload.get("edges")
    if not isinstance(nodes_raw, list) or not isinstance(edges_raw, list):
        raise ValidationError("Lineage payload malformed")
    nodes = [ArtifactLineageNode(**node) for node in nodes_raw]
    edges = [ArtifactLineageEdge(**edge) for edge in edges_raw]
    return ArtifactLineageGraph(nodes=nodes, edges=edges)


def build_lineage_store_from_env() -> LineageStore:
    if os.environ.get("AWS_SAM_LOCAL"):
        base_dir = Path(
            os.environ.get("ARTIFACT_LINEAGE_DIR", DEFAULT_LINEAGE_DIR)
        )
        return LocalLineageStore(base_dir)

    bucket = os.environ.get("ARTIFACT_LINEAGE_BUCKET")
    if bucket:
        prefix = os.environ.get("ARTIFACT_LINEAGE_PREFIX", "lineage")
        return S3LineageStore(bucket=bucket, prefix=prefix)

    base_dir = Path(
        os.environ.get("ARTIFACT_LINEAGE_DIR", DEFAULT_LINEAGE_DIR)
    )
    return LocalLineageStore(base_dir)
