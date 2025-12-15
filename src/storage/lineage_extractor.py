"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Extract lineage graph from HuggingFace model metadata.

This module discovers parent models from Hugging Face metadata
(card_data, config) and README hints (e.g., "base model",
"fine-tuned from") and builds an ArtifactLineageGraph for storage.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Mapping, Optional, Protocol
from urllib.parse import urlparse

from src.clients.hf_client import HFClient
from src.models.lineage import (ArtifactLineageEdge, ArtifactLineageGraph,
                                ArtifactLineageNode)

_LOGGER = logging.getLogger(__name__)

# Defensive caps to avoid explosion
MAX_PARENT_FANOUT = 10

# Regex patterns for README parsing
_PARENT_HINT = re.compile(
    r"(?im)(?:base|parent)"
    r"\s*model[^:\n]*[:]\s*"
    r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
)

_FINE_TUNED_FROM = re.compile(
    r"(?im)fine[- ]tuned\s+(?:from|of|on)\s+([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
)

_HF_LINK = re.compile(
    r"https?://huggingface\.co/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)"
)


class HFClientLike(Protocol):
    """
    get_model_info: Function description.
    :param hf_url:
    :returns:
    """

    """
    HFClientLike: Class description.
    """

    def get_model_info(self, hf_url: str) -> Any: ...

    """
    get_model_readme: Function description.
    :param hf_url:
    :returns:
    """

    def get_model_readme(self, hf_url: str) -> str: ...


def extract_lineage_graph(
    artifact_id: str, source_url: str
) -> ArtifactLineageGraph:
    """Extract lineage graph for a model artifact.

    Args:
        artifact_id: The ID of the main artifact.
        source_url: The HuggingFace URL of the model.

    Returns:
        ArtifactLineageGraph with nodes and edges representing lineage.
        At minimum, returns a graph with just the main artifact node.
        Includes transitive lineage (parent's parents, etc.)
    """
    try:
        hf = HFClient()
    except Exception as exc:  # noqa: BLE001
        _LOGGER.error(
            "Failed to initialize Hugging Face client: %s",
            exc,
            exc_info=True,
        )
        main_name = _extract_name_from_url(source_url)
        main_node = ArtifactLineageNode(
            artifact_id=artifact_id,
            name=main_name,
            source="primary",
        )
        return ArtifactLineageGraph(nodes=[main_node], edges=[])

    # Build complete lineage recursively
    all_nodes: dict[str, ArtifactLineageNode] = {}
    all_edges: list[ArtifactLineageEdge] = []
    visited: set[str] = set()

    try:
        _build_lineage_recursive(
            hf=hf,
            artifact_id=artifact_id,
            source_url=source_url,
            source_type="primary",
            all_nodes=all_nodes,
            all_edges=all_edges,
            visited=visited,
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.error(
            "Failed to extract lineage for %s from %s: %s",
            artifact_id,
            source_url,
            exc,
            exc_info=True,
        )
        # Return minimal graph with just main node
        main_name = _extract_name_from_url(source_url)
        main_node = ArtifactLineageNode(
            artifact_id=artifact_id,
            name=main_name,
            source="primary",
        )
        return ArtifactLineageGraph(nodes=[main_node], edges=[])

    return ArtifactLineageGraph(
        nodes=list(all_nodes.values()),
        edges=all_edges,
    )


def _build_lineage_recursive(
    hf: HFClientLike,
    artifact_id: str,
    source_url: str,
    source_type: str,
    all_nodes: dict[str, ArtifactLineageNode],
    all_edges: list[ArtifactLineageEdge],
    visited: set[str],
    depth: int = 0,
) -> None:
    """Recursively build lineage graph by following parent relationships.

    Args:
        hf: HuggingFace client
        artifact_id: Current artifact ID
        source_url: HuggingFace URL for current artifact
        source_type: Source type for this node
        all_nodes: Accumulated nodes (modified in-place)
        all_edges: Accumulated edges (modified in-place)
        visited: Set of visited HF URLs to prevent cycles
        depth: Current recursion depth (for logging/debugging)
    """
    # Prevent cycles and limit depth
    if source_url in visited or depth > 10:
        return

    visited.add(source_url)

    # Extract model name for this node
    model_name = _extract_name_from_url(source_url)

    # Create node for current artifact
    current_node = ArtifactLineageNode(
        artifact_id=artifact_id,
        name=model_name,
        source=source_type,
    )
    all_nodes[artifact_id] = current_node

    _LOGGER.info(
        "Processing lineage at depth %d: %s (%s)",
        depth,
        artifact_id,
        source_url,
    )

    # Discover parents
    try:
        parents_with_source = _discover_parents_with_source(hf, source_url)
        _LOGGER.info(
            "Found %d parent(s) for %s at depth %d",
            len(parents_with_source),
            artifact_id,
            depth,
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning(
            "Failed to discover parents for %s: %s",
            source_url,
            exc,
        )
        return

    # Process each parent recursively
    for parent_slug, parent_source_type in parents_with_source:
        # Generate deterministic ID for parent
        parent_id = _generate_parent_id(parent_slug)

        # Create edge from current to parent (child -> parent)
        relationship = _determine_relationship(parent_source_type)
        edge = ArtifactLineageEdge(
            from_node_artifact_id=artifact_id,
            to_node_artifact_id=parent_id,
            relationship=relationship,
        )
        all_edges.append(edge)

        # Build HuggingFace URL for parent
        parent_url = f"https://huggingface.co/{parent_slug}"

        # Recursively process parent
        _build_lineage_recursive(
            hf=hf,
            artifact_id=parent_id,
            source_url=parent_url,
            source_type=parent_source_type,
            all_nodes=all_nodes,
            all_edges=all_edges,
            visited=visited,
            depth=depth + 1,
        )


def _discover_parents_with_source(
    hf: HFClientLike, hf_url: str
) -> list[tuple[str, str]]:
    """Discover parent model slugs with their source type.

    Returns:
        List of (parent_slug, source_type) tuples.
        source_type is one of: "config_json", "card_data", "readme"
    """
    parents: list[tuple[str, str]] = []

    # Try metadata first
    try:
        info = hf.get_model_info(hf_url)
    except Exception:
        info = None

    if info is not None:
        card = getattr(info, "card_data", None)
        config = getattr(info, "config", None)

        # Check config first
        if isinstance(config, dict):
            for slug in _extract_parents_from_mapping(config):
                parents.append((slug, "config_json"))
        elif config is not None:
            for slug in _extract_parents_from_object(config):
                parents.append((slug, "config_json"))

        # Check card_data
        if isinstance(card, dict):
            for slug in _extract_parents_from_mapping(card):
                parents.append((slug, "card_data"))
        elif card is not None:
            for slug in _extract_parents_from_object(card):
                parents.append((slug, "card_data"))

        # Check info object itself
        for slug in _extract_parents_from_object(info):
            parents.append((slug, "card_data"))

    # Fallback: scan README for explicit parent/base hints and HF links
    try:
        readme = hf.get_model_readme(hf_url)
    except Exception:
        readme = ""

    for slug in _extract_parents_from_readme(readme or ""):
        parents.append((slug, "readme"))

    # Normalize and dedupe; keep order
    seen: set[str] = set()
    normalized: list[tuple[str, str]] = []
    for item, source_type in parents:
        normalized_slug = _to_repo_slug(item)
        if normalized_slug is not None and normalized_slug not in seen:
            seen.add(normalized_slug)
            normalized.append((normalized_slug, source_type))

    # Limit fan-out defensively
    if len(normalized) > MAX_PARENT_FANOUT:
        _LOGGER.warning(
            "Found %d parents but limiting to %d for %s",
            len(normalized),
            MAX_PARENT_FANOUT,
            hf_url,
        )
    _LOGGER.info(
        "Discovered %d parent(s) for %s: %s",
        len(normalized[:MAX_PARENT_FANOUT]),
        hf_url,
        [slug for slug, _ in normalized[:MAX_PARENT_FANOUT]],
    )
    return normalized[:MAX_PARENT_FANOUT]


def _extract_parents_from_mapping(mapping: Mapping[str, Any]) -> list[str]:
    """Extract parent model slugs from a dictionary-like structure."""
    results: list[str] = []
    params = (
        "base_model",
        "base_model_id",
        "parent_model",
        "parents",
        "parent_models",
        "fine_tuned_from",
        "finetuned_from",
        "adapter_model",
    )
    for key in params:
        value = mapping.get(key)
        if isinstance(value, str):
            results.append(value)
        elif isinstance(value, (list, tuple)):
            results.extend(
                [str(v) for v in value if isinstance(v, (str, bytes))]
            )
    return results


def _extract_parents_from_object(obj: Any) -> list[str]:
    """Extract parent model slugs from an object with attributes."""
    results: list[str] = []
    for attr in (
        "base_model",
        "base_model_id",
        "parent_model",
        "parents",
        "parent_models",
        "fine_tuned_from",
        "finetuned_from",
        "adapter_model",
    ):
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            if isinstance(value, str):
                results.append(value)
            elif isinstance(value, (list, tuple)):
                results.extend(
                    [str(v) for v in value if isinstance(v, (str, bytes))]
                )
    return results


def _extract_parents_from_readme(text: str) -> list[str]:
    """Extract parent model slugs from README text."""
    results: list[str] = []
    for pattern in (_PARENT_HINT, _FINE_TUNED_FROM):
        for m in pattern.finditer(text):
            results.append(m.group(1))
    # Note: _HF_LINK extraction removed to avoid false positives
    # from unrelated links in README (See Also sections, etc.)
    return results


def _to_repo_slug(value: str) -> Optional[str]:
    """Normalize a value to an owner/repo slug."""
    s = (value or "").strip().strip("/")
    if not s:
        return None
    # Convert full URLs to owner/name
    if s.startswith("http://") or s.startswith("https://"):
        try:
            return HFClient._normalize_repo_id(s)
        except Exception:
            return None
    if "/" in s:
        parts = [p for p in s.split("/") if p]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    return None


def _extract_name_from_url(url: str) -> str:
    """Extract model name from HuggingFace URL."""
    try:
        repo_id = HFClient._normalize_repo_id(url)
        if "/" in repo_id:
            return repo_id.split("/")[-1]
        return repo_id
    except Exception:
        # Fallback: extract last path segment
        parsed = urlparse(url)
        segments = [s for s in parsed.path.split("/") if s]
        if segments:
            return segments[-1]
        return "unknown"


def _generate_parent_id(slug: str) -> str:
    """Generate a deterministic artifact ID for a parent model slug.

    Uses a hash of the slug to ensure the same parent always gets the same ID.
    The ID format matches the artifact ID regex: [a-zA-Z0-9-]+
    """
    hash_hex = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:16]
    return f"parent-{hash_hex}"


def _determine_relationship(source_type: str) -> str:
    """Determine the relationship type based on source."""
    if source_type == "readme":
        return "fine_tuned_from"
    return "base_model"
