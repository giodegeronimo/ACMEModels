"""

ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

"""
from __future__ import annotations

from datetime import datetime

import pytest

from src.models import (ArtifactLineageEdge, ArtifactLineageGraph,
                        ArtifactLineageNode, ArtifactMetadata, ArtifactType,
                        User)


def test_artifact_metadata_rejects_empty_id_and_name() -> None:
    """
    test_artifact_metadata_rejects_empty_id_and_name: Function description.
    :param:
    :returns:
    """

    with pytest.raises(ValueError, match="Artifact name cannot be empty"):
        ArtifactMetadata(name="", id="abc123", type=ArtifactType.MODEL)

    with pytest.raises(ValueError, match="Artifact ID cannot be empty"):
        ArtifactMetadata(name="name", id="", type=ArtifactType.MODEL)


def test_artifact_metadata_rejects_wildcard_name() -> None:
    """
    test_artifact_metadata_rejects_wildcard_name: Function description.
    :param:
    :returns:
    """

    with pytest.raises(ValueError, match="Wildcard name '\\*' is only allowed"):
        ArtifactMetadata(name="*", id="abc123", type=ArtifactType.MODEL)


def test_audit_model_validations() -> None:
    """
    test_audit_model_validations: Function description.
    :param:
    :returns:
    """

    with pytest.raises(ValueError, match="User name cannot be empty"):
        User(name="", is_admin=False)

    # Naive timestamps are rejected.
    from src.models.audit import _ensure_utc

    with pytest.raises(
        ValueError,
        match="Audit timestamps must include timezone information",
    ):
        _ensure_utc(datetime.utcnow())


def test_lineage_model_validations() -> None:
    """
    test_lineage_model_validations: Function description.
    :param:
    :returns:
    """

    with pytest.raises(ValueError, match="Lineage relationship label cannot be empty"):
        ArtifactLineageEdge(
            from_node_artifact_id="a",
            to_node_artifact_id="b",
            relationship="",
        )

    with pytest.raises(ValueError, match="Lineage graph must include at least one node"):
        ArtifactLineageGraph(nodes=[], edges=[])

    # Still allows building valid graphs.
    node = ArtifactLineageNode(artifact_id="abc123", name="main")
    assert ArtifactLineageGraph(nodes=[node], edges=[]).nodes[0].artifact_id == "abc123"
