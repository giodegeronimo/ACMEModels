"""Extract lineage graph from HuggingFace model metadata.

This module discovers parent models from Hugging Face metadata (card_data, config)
and README hints (e.g., "base model", "fine-tuned from") and builds an
ArtifactLineageGraph for storage.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

from src.clients.hf_client import HFClient
from src.models.lineage import (
    ArtifactLineageEdge,
    ArtifactLineageGraph,
    ArtifactLineageNode,
)

_LOGGER = logging.getLogger(__name__)

# Defensive caps to avoid explosion
MAX_PARENT_FANOUT = 5

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
    """
    # Extract model name from URL for the main node
    main_name = _extract_name_from_url(source_url)

    # Create the main artifact node
    main_node = ArtifactLineageNode(
        artifact_id=artifact_id,
        name=main_name,
        source="primary",
    )

    nodes: list[ArtifactLineageNode] = [main_node]
    edges: list[ArtifactLineageEdge] = []

    try:
        # Discover parents using HF API
        hf = HFClient()
        parents_with_source = _discover_parents_with_source(hf, source_url)

        for parent_slug, source_type in parents_with_source:
            # Generate deterministic ID for parent
            parent_id = _generate_parent_id(parent_slug)
            parent_name = (
                parent_slug.split("/")[-1] if "/" in parent_slug else parent_slug
            )

            parent_node = ArtifactLineageNode(
                artifact_id=parent_id,
                name=parent_name,
                source=source_type,
            )
            nodes.append(parent_node)

            # Edge from main artifact to parent (child -> parent)
            relationship = _determine_relationship(source_type)
            edge = ArtifactLineageEdge(
                from_node_artifact_id=artifact_id,
                to_node_artifact_id=parent_id,
                relationship=relationship,
            )
            edges.append(edge)

    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning(
            "Failed to extract lineage parents for %s: %s",
            artifact_id,
            exc,
        )
        # Return minimal graph with just main node
        return ArtifactLineageGraph(nodes=[main_node], edges=[])

    return ArtifactLineageGraph(nodes=nodes, edges=edges)


def _discover_parents_with_source(
    hf: HFClient, hf_url: str
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
        slug = _to_repo_slug(item)
        if slug and slug not in seen:
            seen.add(slug)
            normalized.append((slug, source_type))

    # Limit fan-out defensively
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
    for attr in ("base_model", "parent_model", "parents", "parent_models"):
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
