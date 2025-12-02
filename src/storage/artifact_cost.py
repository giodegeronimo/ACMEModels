"""Artifact cost calculation utilities."""

from __future__ import annotations

import os
from typing import Any

from src.models.artifacts import validate_artifact_id
from src.models.lineage import ArtifactLineageGraph
from src.storage.blob_store import BlobNotFoundError


class CostCalculationError(RuntimeError):
    """Raised when cost calculation fails."""


def calculate_artifact_cost(
    artifact_id: str,
    *,
    include_dependencies: bool = False,
    lineage_graph: ArtifactLineageGraph | None = None,
) -> dict[str, dict[str, float]]:
    """
    Calculate the cost (size in MB) of an artifact and optionally its
    dependencies.

    Args:
        artifact_id: The artifact to calculate cost for
        include_dependencies: Whether to include dependency costs
        lineage_graph: Lineage graph for finding dependencies
            (required if include_dependencies=True)

    Returns:
        Dictionary mapping artifact_id -> cost info:
        {
            "artifact_id": {
                "standalone_cost": 412.5,  # always included
                "total_cost": 1255.0  # sum of artifact + dependencies (if any)
            }
        }
    """
    validate_artifact_id(artifact_id)

    # Get S3 client
    bucket = os.environ.get("ARTIFACT_STORAGE_BUCKET")
    if not bucket:
        raise CostCalculationError("ARTIFACT_STORAGE_BUCKET not configured")

    prefix = os.environ.get("ARTIFACT_STORAGE_PREFIX", "artifacts")

    try:
        import boto3
    except ImportError as exc:
        raise CostCalculationError(
            "boto3 is required for cost calculation"
        ) from exc

    region = (
        os.environ.get("ARTIFACT_STORAGE_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )
    client_kwargs: dict[str, Any] = {}
    if region:
        client_kwargs["region_name"] = region

    s3 = boto3.client("s3", **client_kwargs)

    # Calculate standalone cost for the main artifact
    standalone_cost = _get_artifact_size_mb(s3, bucket, prefix, artifact_id)

    if not include_dependencies:
        # Simple case: return standalone_cost = total_cost (no dependencies)
        return {
            artifact_id: {
                "standalone_cost": standalone_cost,
                "total_cost": standalone_cost
            }
        }

    # Complex case: calculate all dependencies
    if lineage_graph is None:
        raise CostCalculationError(
            "Lineage graph required when include_dependencies=True"
        )

    # Collect all dependency artifact IDs from the lineage graph
    dependency_ids = _extract_dependency_ids(lineage_graph, artifact_id)

    # Calculate costs for all dependencies
    result: dict[str, dict[str, float]] = {}

    # Add main artifact
    total_cost_sum = standalone_cost
    result[artifact_id] = {
        "standalone_cost": standalone_cost,
        "total_cost": 0.0,  # Will be set after summing all dependencies
    }

    # Add each dependency
    for dep_id in dependency_ids:
        dep_cost = _get_artifact_size_mb(s3, bucket, prefix, dep_id)
        result[dep_id] = {
            "standalone_cost": dep_cost,
            "total_cost": dep_cost,
        }
        total_cost_sum += dep_cost

    # Set the total_cost for the main artifact
    result[artifact_id]["total_cost"] = total_cost_sum

    return result


def _get_artifact_size_mb(
    s3_client: Any,
    bucket: str,
    prefix: str,
    artifact_id: str,
) -> float:
    """Get the size of an artifact in MB from S3."""
    key = f"{prefix}/{artifact_id}" if prefix else artifact_id

    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        size_bytes = response.get("ContentLength", 0)
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 2)
    except s3_client.exceptions.NoSuchKey as exc:
        raise BlobNotFoundError(
            f"Artifact '{artifact_id}' not found in S3"
        ) from exc
    except Exception as exc:
        raise CostCalculationError(
            f"Failed to get size for artifact '{artifact_id}': {exc}"
        ) from exc


def _extract_dependency_ids(
    lineage_graph: ArtifactLineageGraph,
    root_artifact_id: str,
) -> set[str]:
    """
    Extract all dependency artifact IDs from the lineage graph.

    Dependencies are nodes that have edges pointing TO the root artifact.
    (i.e., from_node -> root_artifact)
    """
    dependencies = set()

    for edge in lineage_graph.edges:
        # If this edge points to our root artifact, the source is a dependency
        if edge.to_node_artifact_id == root_artifact_id:
            dependencies.add(edge.from_node_artifact_id)

    return dependencies
