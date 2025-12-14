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


def store_lineage(
    artifact_id: str, lineage_graph: ArtifactLineageGraph
) -> None:
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
        raise RuntimeError(
            "MODEL_RESULTS_BUCKET is not configured"
        )
    # Use dedicated lineage prefix, not MODEL_RESULTS_PREFIX
    # (which is for ratings)
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
        lineage_dict = json.loads(
            path.read_text(encoding="utf-8")
        )
    else:
        bucket = os.environ.get("MODEL_RESULTS_BUCKET")
        if not bucket:
            return None
        # Use dedicated lineage prefix, not MODEL_RESULTS_PREFIX
    # (which is for ratings)
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


def load_complete_lineage_family(
    artifact_id: str,
) -> ArtifactLineageGraph | None:
    """Load complete lineage graph including all related artifacts.

    For a given artifact, this finds ALL artifacts that share lineage
    (both parents and children) and merges them into a unified graph.

    This implements the requirement that "all three requests must produce
    the same graph" - querying any artifact in a lineage family returns
    the complete family tree.

    Args:
        artifact_id: The artifact to query lineage for

    Returns:
        Complete merged lineage graph, or None if no lineage found
    """
    # First, load the direct lineage for this artifact
    direct_lineage = load_lineage(artifact_id)
    if direct_lineage is None:
        return None

    # Collect all artifact IDs AND names in this lineage family
    family_ids: set[str] = {artifact_id}
    family_names: set[str] = set()
    for node in direct_lineage.nodes:
        family_ids.add(node.artifact_id)
        if node.name:
            family_names.add(node.name)

    # Now find ALL lineage graphs that reference any of these IDs or names
    # This captures children that have this artifact as a parent
    all_graphs = _find_all_related_lineage_graphs(family_ids, family_names)

    # Merge all graphs into one
    return _merge_lineage_graphs(all_graphs)


def _find_all_related_lineage_graphs(
    seed_ids: set[str],
    seed_names: set[str],
) -> list[ArtifactLineageGraph]:
    """Find all lineage graphs that contain any of the seed IDs or names.

    This scans ALL stored lineage files and returns those that contain
    any of the seed IDs OR names in their nodes. This captures both direct
    lineage and reverse relationships (children).

    Args:
        seed_ids: Initial set of artifact IDs to search for
        seed_names: Initial set of model names to search for

    Returns:
        List of all related lineage graphs
    """
    all_graphs: list[ArtifactLineageGraph] = []
    related_artifact_ids: set[str] = set()
    related_names: set[str] = seed_names.copy()

    # Get all lineage files
    lineage_files = _list_all_lineage_files()

    # First pass: find all graphs that contain any seed ID or name
    for file_artifact_id in lineage_files:
        graph = load_lineage(file_artifact_id)
        if graph is None:
            continue

        # Check if this graph contains any seed ID or name
        graph_node_ids = {node.artifact_id for node in graph.nodes}
        graph_node_names = {node.name for node in graph.nodes if node.name}

        if (graph_node_ids & seed_ids) or (graph_node_names & seed_names):
            all_graphs.append(graph)
            related_artifact_ids.update(graph_node_ids)
            related_names.update(graph_node_names)

    # Second pass: find graphs that contain any newly discovered IDs or names
    # This catches transitive relationships
    added_files = {
        next((n.artifact_id for n in g.nodes if n.source == "primary"), None)
        for g in all_graphs
    }

    for file_artifact_id in lineage_files:
        if file_artifact_id in added_files:
            continue  # Already added

        graph = load_lineage(file_artifact_id)
        if graph is None:
            continue

        graph_node_ids = {node.artifact_id for node in graph.nodes}
        graph_node_names = {node.name for node in graph.nodes if node.name}

        has_related_id = graph_node_ids & related_artifact_ids
        has_related_name = graph_node_names & related_names
        if has_related_id or has_related_name:
            all_graphs.append(graph)
            added_files.add(file_artifact_id)

    return all_graphs


def _list_all_lineage_files() -> list[str]:
    """List all artifact IDs that have lineage files.

    Returns:
        List of artifact IDs (file basenames without .json)
    """
    if os.environ.get("AWS_SAM_LOCAL"):
        if not _LOCAL_LINEAGE_DIR.exists():
            return []
        return [
            p.stem
            for p in _LOCAL_LINEAGE_DIR.iterdir()
            if p.suffix == ".json"
        ]

    bucket = os.environ.get("MODEL_RESULTS_BUCKET")
    if not bucket:
        return []

    prefix = "lineage/"
    client = _build_s3_client()

    try:
        artifact_ids: list[str] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Extract artifact ID from key: lineage/{id}.json
                if key.startswith(prefix) and key.endswith(".json"):
                    # Remove prefix and .json suffix
                    artifact_id = key[len(prefix):-5]
                    artifact_ids.append(artifact_id)
        return artifact_ids
    except Exception as exc:  # pragma: no cover
        _LOGGER.error("Failed to list lineage files: %s", exc)
        return []


def _merge_lineage_graphs(
    graphs: list[ArtifactLineageGraph],
) -> ArtifactLineageGraph:
    """Merge multiple lineage graphs into one.

    Deduplicates nodes by NAME (not artifact_id) and remaps edges.
    This handles the case where the same model appears with
    different IDs (e.g., as a real artifact vs as a parent-*
    synthetic ID).

    Args:
        graphs: List of lineage graphs to merge

    Returns:
        Single merged lineage graph with nodes deduplicated by name
    """
    from src.models.lineage import ArtifactLineageEdge, ArtifactLineageNode

    # Deduplicate nodes by name, preferring primary sources
    nodes_by_name: dict[str, ArtifactLineageNode] = {}
    nodes_by_id: dict[str, ArtifactLineageNode] = {}  # For unnamed nodes
    id_to_canonical_id: dict[str, str] = {}  # Map all IDs to canonical ID

    for graph in graphs:
        for node in graph.nodes:
            name = node.name
            if name is None:
                # Deduplicate unnamed nodes by artifact_id
                if node.artifact_id not in nodes_by_id:
                    nodes_by_id[node.artifact_id] = node
                id_to_canonical_id[node.artifact_id] = node.artifact_id
                continue
            if name not in nodes_by_name:
                # First occurrence of this name
                nodes_by_name[name] = node
                id_to_canonical_id[node.artifact_id] = node.artifact_id
            else:
                # Prefer "primary" source over synthetic parent IDs
                existing = nodes_by_name[name]
                if node.source == "primary" and existing.source != "primary":
                    # Replace with primary version
                    old_canonical_id = existing.artifact_id
                    nodes_by_name[name] = node
                    # Update all mappings
                    for k, v in list(id_to_canonical_id.items()):
                        if v == old_canonical_id:
                            id_to_canonical_id[k] = node.artifact_id
                else:
                    # Map this ID to the existing canonical ID
                    id_to_canonical_id[node.artifact_id] = existing.artifact_id

    # Deduplicate and remap edges
    edge_set: set[tuple[str, str, str]] = set()
    edges: list[ArtifactLineageEdge] = []

    for graph in graphs:
        for edge in graph.edges:
            # Map edge IDs to canonical IDs
            from_id = id_to_canonical_id.get(
                edge.from_node_artifact_id,
                edge.from_node_artifact_id,
            )
            to_id = id_to_canonical_id.get(
                edge.to_node_artifact_id,
                edge.to_node_artifact_id,
            )

            edge_tuple = (from_id, to_id, edge.relationship)
            if edge_tuple not in edge_set:
                edge_set.add(edge_tuple)
                edges.append(
                    ArtifactLineageEdge(
                        from_node_artifact_id=from_id,
                        to_node_artifact_id=to_id,
                        relationship=edge.relationship,
                    )
                )

    # Combine named nodes and unnamed nodes
    all_nodes = list(nodes_by_name.values()) + list(nodes_by_id.values())

    return ArtifactLineageGraph(
        nodes=all_nodes,
        edges=edges,
    )
