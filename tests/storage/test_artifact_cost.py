"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Unit tests for artifact cost helpers.
"""

from __future__ import annotations

import types
from typing import Any, Dict, Optional

import pytest

from src.models.lineage import (ArtifactLineageEdge, ArtifactLineageGraph,
                                ArtifactLineageNode)
from src.storage.artifact_cost import (CostCalculationError,
                                       calculate_artifact_cost)
from src.storage.blob_store import BlobNotFoundError


class _FakeS3:
    """
    _FakeS3: Class description.
    """

    class exceptions:
        """
        exceptions: Class description.
        """

        class NoSuchKey(Exception):
            """
            NoSuchKey: Class description.
            """

            pass

    def __init__(self) -> None:
        """
        __init__: Function description.
        :param:
        :returns:
        """

        self.sizes: dict[tuple[str, str], int] = {}
        self.raise_on_head: Optional[Exception] = None

    def head_object(self, *, Bucket: str, Key: str) -> Dict[str, Any]:
        """
        head_object: Function description.
        :param Bucket:
        :param Key:
        :returns:
        """

        if self.raise_on_head is not None:
            raise self.raise_on_head
        if (Bucket, Key) not in self.sizes:
            raise self.exceptions.NoSuchKey("missing")
        return {"ContentLength": self.sizes[(Bucket, Key)]}


def _graph_with_dependencies(root: str, deps: list[str]) -> ArtifactLineageGraph:
    """
    _graph_with_dependencies: Function description.
    :param root:
    :param deps:
    :returns:
    """

    nodes = [ArtifactLineageNode(artifact_id=root, name=root)]
    nodes.extend(ArtifactLineageNode(artifact_id=dep, name=dep) for dep in deps)
    edges = [
        ArtifactLineageEdge(
            from_node_artifact_id=root,
            to_node_artifact_id=dep,
            relationship="depends_on",
        )
        for dep in deps
    ]
    return ArtifactLineageGraph(nodes=nodes, edges=edges)


def test_calculate_artifact_cost_requires_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_calculate_artifact_cost_requires_bucket: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.delenv("ARTIFACT_STORAGE_BUCKET", raising=False)

    with pytest.raises(CostCalculationError):
        calculate_artifact_cost("abc123")


def test_calculate_artifact_cost_standalone(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_calculate_artifact_cost_standalone: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("ARTIFACT_STORAGE_BUCKET", "bucket")
    monkeypatch.setenv("ARTIFACT_STORAGE_PREFIX", "artifacts")
    fake_s3 = _FakeS3()
    fake_s3.sizes[("bucket", "artifacts/abc123")] = 1024 * 1024  # 1 MB
    monkeypatch.setitem(
        __import__("sys").modules,
        "boto3",
        types.SimpleNamespace(client=lambda *_args, **_kwargs: fake_s3),
    )

    result = calculate_artifact_cost("abc123")

    assert result == {"abc123": {"standalone_cost": 1.0, "total_cost": 1.0}}


def test_calculate_artifact_cost_with_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_calculate_artifact_cost_with_dependencies: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("ARTIFACT_STORAGE_BUCKET", "bucket")
    monkeypatch.setenv("ARTIFACT_STORAGE_PREFIX", "artifacts")
    fake_s3 = _FakeS3()
    fake_s3.sizes[("bucket", "artifacts/root")] = 1024 * 1024
    fake_s3.sizes[("bucket", "artifacts/dep1")] = 2 * 1024 * 1024
    fake_s3.sizes[("bucket", "artifacts/dep2")] = 1024 * 1024
    monkeypatch.setitem(
        __import__("sys").modules,
        "boto3",
        types.SimpleNamespace(client=lambda *_args, **_kwargs: fake_s3),
    )

    graph = _graph_with_dependencies("root", ["dep1", "dep2"])
    result = calculate_artifact_cost(
        "root",
        include_dependencies=True,
        lineage_graph=graph,
    )

    assert result["root"]["standalone_cost"] == 1.0
    assert result["root"]["total_cost"] == 4.0
    assert result["dep1"]["standalone_cost"] == 2.0


def test_calculate_artifact_cost_include_deps_requires_lineage_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_calculate_artifact_cost_include_deps_requires_lineage_graph: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("ARTIFACT_STORAGE_BUCKET", "bucket")
    monkeypatch.setenv("ARTIFACT_STORAGE_PREFIX", "artifacts")
    fake_s3 = _FakeS3()
    fake_s3.sizes[("bucket", "artifacts/abc123")] = 1024 * 1024
    monkeypatch.setitem(
        __import__("sys").modules,
        "boto3",
        types.SimpleNamespace(client=lambda *_args, **_kwargs: fake_s3),
    )

    with pytest.raises(CostCalculationError):
        calculate_artifact_cost("abc123", include_dependencies=True)


def test_calculate_artifact_cost_missing_object_raises_blob_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_calculate_artifact_cost_missing_object_raises_blob_not_found: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("ARTIFACT_STORAGE_BUCKET", "bucket")
    fake_s3 = _FakeS3()
    monkeypatch.setitem(
        __import__("sys").modules,
        "boto3",
        types.SimpleNamespace(client=lambda *_args, **_kwargs: fake_s3),
    )

    with pytest.raises(BlobNotFoundError):
        calculate_artifact_cost("abc123")


def test_calculate_artifact_cost_raises_when_boto3_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_calculate_artifact_cost_raises_when_boto3_missing: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("ARTIFACT_STORAGE_BUCKET", "bucket")

    import builtins

    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        """
        fake_import: Function description.
        :param name:
        :param *args:
        :param **kwargs:
        :returns:
        """

        if name == "boto3":
            raise ImportError("no boto3")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(CostCalculationError, match="boto3 is required for cost calculation"):
        calculate_artifact_cost("abc123")
