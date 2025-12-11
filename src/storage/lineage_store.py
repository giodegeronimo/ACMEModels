"""Persistent storage helpers for artifact lineage graphs."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from src.models.lineage import ArtifactLineageGraph

try:  # pragma: no cover
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]
    Config = None  # type: ignore[assignment]

    class ClientError(Exception):  # type: ignore[no-redef]
        """Placeholder when botocore is unavailable."""

    class BotoCoreError(Exception):  # type: ignore[no-redef]
        """Placeholder when botocore is unavailable."""

_LOCAL_LINEAGE_DIR = Path(
    os.environ.get("ARTIFACT_LINEAGE_DIR", "/tmp/acme-artifact-lineage")
)
_LOGGER = logging.getLogger(__name__)
_S3_CLIENT = None


class LineageStoreError(RuntimeError):
    """Raised when lineage persistence fails."""


def _build_s3_client() -> Any:
    global _S3_CLIENT
    if _S3_CLIENT is not None:
        return _S3_CLIENT

    if boto3 is None:  # pragma: no cover - boto3 always present in prod
        raise LineageStoreError("boto3 is required for lineage storage")

    region = (
        os.environ.get("ARTIFACT_STORAGE_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )
    client_kwargs: Dict[str, Any] = {}

    if region:
        client_kwargs["region_name"] = region

    if Config is not None:
        cfg = Config(
            retries={"max_attempts": 5, "mode": "standard"},
        )
        client_kwargs["config"] = cfg

    _S3_CLIENT = boto3.client("s3", **client_kwargs)
    return _S3_CLIENT


def store_lineage(artifact_id: str, lineage_graph: ArtifactLineageGraph) -> None:
    """Store lineage graph to S3 or local filesystem."""
    # Serialize to dict
    lineage_dict = {
        "nodes": [
            {
                "artifact_id": node.artifact_id,
                "name": node.name,
                "source": node.source,
                **({"metadata": node.metadata} if node.metadata else {}),
            }
            for node in lineage_graph.nodes
        ],
        "edges": [
            {
                "from_node_artifact_id": edge.from_node_artifact_id,
                "to_node_artifact_id": edge.to_node_artifact_id,
                "relationship": edge.relationship,
            }
            for edge in lineage_graph.edges
        ],
    }

    if os.environ.get("AWS_SAM_LOCAL"):
        _LOCAL_LINEAGE_DIR.mkdir(parents=True, exist_ok=True)
        path = _LOCAL_LINEAGE_DIR / f"{artifact_id}.json"
        path.write_text(json.dumps(lineage_dict), encoding="utf-8")
        return

    bucket = os.environ.get("MODEL_RESULTS_BUCKET")
    if not bucket:
        raise RuntimeError("MODEL_RESULTS_BUCKET is not configured")
    # Use dedicated lineage prefix, not MODEL_RESULTS_PREFIX (which is for ratings)
    prefix = "lineage"
    key = f"{prefix}/{artifact_id}.json"
    client = _build_s3_client()
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(lineage_dict).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as exc:  # pragma: no cover - network errors
        _LOGGER.error(
            "Failed to store lineage for artifact_id=%s: %s",
            artifact_id,
            exc,
        )
        raise LineageStoreError(
            f"Failed to store lineage for '{artifact_id}'"
        ) from exc


def load_lineage(artifact_id: str) -> ArtifactLineageGraph | None:
    """Load lineage graph from S3 or local filesystem."""
    if os.environ.get("AWS_SAM_LOCAL"):
        path = _LOCAL_LINEAGE_DIR / f"{artifact_id}.json"
        if not path.exists():
            return None
        lineage_dict = json.loads(path.read_text(encoding="utf-8"))
    else:
        bucket = os.environ.get("MODEL_RESULTS_BUCKET")
        if not bucket:
            return None
        # Use dedicated lineage prefix, not MODEL_RESULTS_PREFIX (which is for ratings)
        prefix = "lineage"
        key = f"{prefix}/{artifact_id}.json"
        client = _build_s3_client()
        try:
            response = client.get_object(Bucket=bucket, Key=key)
        except ClientError as exc:  # pragma: no cover - requires network
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code == "NoSuchKey":
                return None
            _LOGGER.error(
                "ClientError fetching lineage for artifact_id=%s: %s",
                artifact_id,
                exc,
            )
            raise LineageStoreError(
                f"Failed to load lineage for '{artifact_id}'"
            ) from exc
        except BotoCoreError as exc:  # pragma: no cover
            _LOGGER.error(
                "BotoCoreError fetching lineage for artifact_id=%s: %s",
                artifact_id,
                exc,
            )
            raise LineageStoreError(
                f"Failed to load lineage for '{artifact_id}'"
            ) from exc
        lineage_dict = json.loads(response["Body"].read())

    # Deserialize from dict
    from src.models.lineage import ArtifactLineageEdge, ArtifactLineageNode

    nodes = [
        ArtifactLineageNode(
            artifact_id=node["artifact_id"],
            name=node["name"],
            source=node["source"],
            metadata=node.get("metadata"),
        )
        for node in lineage_dict.get("nodes", [])
    ]
    edges = [
        ArtifactLineageEdge(
            from_node_artifact_id=edge["from_node_artifact_id"],
            to_node_artifact_id=edge["to_node_artifact_id"],
            relationship=edge["relationship"],
        )
        for edge in lineage_dict.get("edges", [])
    ]

    return ArtifactLineageGraph(nodes=nodes, edges=edges)
