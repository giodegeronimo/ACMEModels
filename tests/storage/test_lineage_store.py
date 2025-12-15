"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Unit tests for lineage graph persistence helpers.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from src.models.lineage import (ArtifactLineageEdge, ArtifactLineageGraph,
                                ArtifactLineageNode)
from src.storage import lineage_store


def _graph() -> ArtifactLineageGraph:
    """
    _graph: Function description.
    :param:
    :returns:
    """

    return ArtifactLineageGraph(
        nodes=[
            ArtifactLineageNode(
                artifact_id="abc123",
                name="main",
                source="primary",
            ),
            ArtifactLineageNode(artifact_id="parent-1", name="base", source="card"),
        ],
        edges=[
            ArtifactLineageEdge(
                from_node_artifact_id="abc123",
                to_node_artifact_id="parent-1",
                relationship="base_model",
            )
        ],
    )


def test_store_and_load_lineage_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_store_and_load_lineage_local: Function description.
    :param tmp_path:
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("AWS_SAM_LOCAL", "1")
    monkeypatch.setattr(lineage_store, "_LOCAL_LINEAGE_DIR", tmp_path)

    graph = _graph()
    lineage_store.store_lineage("abc123", graph)
    loaded = lineage_store.load_lineage("abc123")

    assert loaded == graph


def test_load_lineage_local_missing_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    test_load_lineage_local_missing_returns_none: Function description.
    :param tmp_path:
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("AWS_SAM_LOCAL", "1")
    monkeypatch.setattr(lineage_store, "_LOCAL_LINEAGE_DIR", tmp_path)

    assert lineage_store.load_lineage("missing") is None


def test_store_lineage_requires_bucket_in_s3_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_store_lineage_requires_bucket_in_s3_mode: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)
    monkeypatch.delenv("MODEL_RESULTS_BUCKET", raising=False)

    with pytest.raises(RuntimeError):
        lineage_store.store_lineage("abc123", _graph())


class _FakeClientError(Exception):
    """
    _FakeClientError: Class description.
    """

    def __init__(self, code: str) -> None:
        """
        __init__: Function description.
        :param code:
        :returns:
        """

        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _FakeS3Client:
    """
    _FakeS3Client: Class description.
    """

    def __init__(self) -> None:
        """
        __init__: Function description.
        :param:
        :returns:
        """

        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_calls: list[tuple[str, str]] = []
        self.get_calls: list[tuple[str, str]] = []
        self.raise_on_get: Optional[Exception] = None

    def put_object(
        self, *, Bucket: str, Key: str, Body: bytes, ContentType: str
    ) -> None:
        """
        put_object: Function description.
        :param Bucket:
        :param Key:
        :param Body:
        :param ContentType:
        :returns:
        """

        self.put_calls.append((Bucket, Key))
        self.objects[(Bucket, Key)] = Body

    def get_object(self, *, Bucket: str, Key: str) -> Dict[str, Any]:
        """
        get_object: Function description.
        :param Bucket:
        :param Key:
        :returns:
        """

        self.get_calls.append((Bucket, Key))
        if self.raise_on_get is not None:
            raise self.raise_on_get
        if (Bucket, Key) not in self.objects:
            raise _FakeClientError("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


def test_store_and_load_lineage_s3_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_store_and_load_lineage_s3_happy_path: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)
    monkeypatch.setenv("MODEL_RESULTS_BUCKET", "bucket")
    client = _FakeS3Client()
    monkeypatch.setattr(lineage_store, "_build_s3_client", lambda: client)
    monkeypatch.setattr(lineage_store, "ClientError", _FakeClientError)

    graph = _graph()
    lineage_store.store_lineage("abc123", graph)
    loaded = lineage_store.load_lineage("abc123")

    assert loaded == graph
    assert client.put_calls == [("bucket", "lineage/abc123.json")]


def test_load_lineage_s3_missing_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_load_lineage_s3_missing_returns_none: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)
    monkeypatch.setenv("MODEL_RESULTS_BUCKET", "bucket")
    client = _FakeS3Client()
    client.raise_on_get = _FakeClientError("NoSuchKey")
    monkeypatch.setattr(lineage_store, "_build_s3_client", lambda: client)
    monkeypatch.setattr(lineage_store, "ClientError", _FakeClientError)

    assert lineage_store.load_lineage("missing") is None


def test_load_lineage_s3_bucket_missing_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_load_lineage_s3_bucket_missing_returns_none: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)
    monkeypatch.delenv("MODEL_RESULTS_BUCKET", raising=False)

    assert lineage_store.load_lineage("abc123") is None


def test_build_s3_client_uses_region_and_config_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_build_s3_client_uses_region_and_config_and_caches: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setattr(lineage_store, "_S3_CLIENT", None)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    captured: list[dict[str, Any]] = []

    class _FakeConfig:
        """
        _FakeConfig: Class description.
        """

        def __init__(self, *, retries: dict[str, object]) -> None:
            """
            __init__: Function description.
            :param retries:
            :returns:
            """

            self.retries = retries

    class _FakeBoto3:
        """
        _FakeBoto3: Class description.
        """

        def client(self, service: str, **kwargs: Any) -> Any:
            """
            client: Function description.
            :param service:
            :param **kwargs:
            :returns:
            """

            captured.append({"service": service, "kwargs": kwargs})
            return object()

    monkeypatch.setattr(lineage_store, "Config", _FakeConfig)
    monkeypatch.setattr(lineage_store, "boto3", _FakeBoto3())

    client1 = lineage_store._build_s3_client()
    client2 = lineage_store._build_s3_client()

    assert client1 is client2
    assert captured and captured[0]["service"] == "s3"
    kwargs = captured[0]["kwargs"]
    assert kwargs["region_name"] == "us-east-1"
    assert isinstance(kwargs["config"], _FakeConfig)


def test_build_s3_client_raises_when_boto3_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_build_s3_client_raises_when_boto3_missing: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setattr(lineage_store, "_S3_CLIENT", None)
    monkeypatch.setattr(lineage_store, "boto3", None)
    with pytest.raises(
        lineage_store.LineageStoreError,
        match="boto3 is required for lineage storage",
    ):
        lineage_store._build_s3_client()


def test_store_lineage_logs_and_raises_on_s3_put_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    test_store_lineage_logs_and_raises_on_s3_put_failure: Function description.
    :param monkeypatch:
    :param caplog:
    :returns:
    """

    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)
    monkeypatch.setenv("MODEL_RESULTS_BUCKET", "bucket")

    class ExplodingClient(_FakeS3Client):
        """
        ExplodingClient: Class description.
        """

        def put_object(self, **kwargs: Any) -> None:  # type: ignore[override]
            """
            put_object: Function description.
            :param **kwargs:
            :returns:
            """

            raise RuntimeError("boom")

    monkeypatch.setattr(lineage_store, "_build_s3_client", lambda: ExplodingClient())

    with caplog.at_level("ERROR"):
        with pytest.raises(lineage_store.LineageStoreError, match="Failed to store lineage"):
            lineage_store.store_lineage("abc123", _graph())

    assert "Failed to store lineage for artifact_id=abc123" in caplog.text


def test_load_lineage_logs_and_raises_on_client_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    test_load_lineage_logs_and_raises_on_client_error: Function description.
    :param monkeypatch:
    :param caplog:
    :returns:
    """

    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)
    monkeypatch.setenv("MODEL_RESULTS_BUCKET", "bucket")

    class LocalClientError(Exception):
        """
        LocalClientError: Class description.
        """

        def __init__(self, code: str) -> None:
            """
            __init__: Function description.
            :param code:
            :returns:
            """

            super().__init__(code)
            self.response = {"Error": {"Code": code}}

    class Client(_FakeS3Client):
        """
        Client: Class description.
        """

        def get_object(self, *, Bucket: str, Key: str) -> Dict[str, Any]:  # type: ignore[override]
            """
            get_object: Function description.
            :param Bucket:
            :param Key:
            :returns:
            """

            raise LocalClientError("Boom")

    monkeypatch.setattr(lineage_store, "ClientError", LocalClientError)
    monkeypatch.setattr(lineage_store, "_build_s3_client", lambda: Client())

    with caplog.at_level("ERROR"):
        with pytest.raises(lineage_store.LineageStoreError, match="Failed to load lineage"):
            lineage_store.load_lineage("abc123")

    assert "ClientError fetching lineage for artifact_id=abc123" in caplog.text


def test_load_lineage_logs_and_raises_on_botocore_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    test_load_lineage_logs_and_raises_on_botocore_error: Function description.
    :param monkeypatch:
    :param caplog:
    :returns:
    """

    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)
    monkeypatch.setenv("MODEL_RESULTS_BUCKET", "bucket")

    class LocalBotoCoreError(Exception):
        """
        LocalBotoCoreError: Class description.
        """

        pass

    class Client(_FakeS3Client):
        """
        Client: Class description.
        """

        def get_object(self, *, Bucket: str, Key: str) -> Dict[str, Any]:  # type: ignore[override]
            """
            get_object: Function description.
            :param Bucket:
            :param Key:
            :returns:
            """

            raise LocalBotoCoreError("boom")

    monkeypatch.setattr(lineage_store, "BotoCoreError", LocalBotoCoreError)
    monkeypatch.setattr(lineage_store, "_build_s3_client", lambda: Client())

    with caplog.at_level("ERROR"):
        with pytest.raises(lineage_store.LineageStoreError, match="Failed to load lineage"):
            lineage_store.load_lineage("abc123")

    assert "BotoCoreError fetching lineage for artifact_id=abc123" in caplog.text


def test_list_all_lineage_files_local_lists_only_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_list_all_lineage_files_local_lists_only_json: Function description.
    :param tmp_path:
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("AWS_SAM_LOCAL", "1")
    local_dir = tmp_path / "lineage"
    monkeypatch.setattr(lineage_store, "_LOCAL_LINEAGE_DIR", local_dir)

    assert lineage_store._list_all_lineage_files() == []

    local_dir.mkdir()
    (local_dir / "a.json").write_text("{}", encoding="utf-8")
    (local_dir / "b.txt").write_text("{}", encoding="utf-8")
    (local_dir / "c.json").write_text("{}", encoding="utf-8")

    assert sorted(lineage_store._list_all_lineage_files()) == ["a", "c"]


def test_list_all_lineage_files_s3_uses_paginator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_list_all_lineage_files_s3_uses_paginator: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)
    monkeypatch.setenv("MODEL_RESULTS_BUCKET", "bucket")

    class _FakePaginator:
        """
        _FakePaginator: Class description.
        """

        def paginate(self, **_: object):
            """
            paginate: Function description.
            :param **_:
            :returns:
            """

            return [
                {"Contents": [{"Key": "lineage/a.json"}, {"Key": "lineage/a.txt"}]},
                {"Contents": [{"Key": "lineage/b.json"}, {"Key": "other/c.json"}]},
                {"Contents": []},
            ]

    class _FakeClient:
        """
        _FakeClient: Class description.
        """

        def get_paginator(self, name: str) -> _FakePaginator:
            """
            get_paginator: Function description.
            :param name:
            :returns:
            """

            assert name == "list_objects_v2"
            return _FakePaginator()

    monkeypatch.setattr(lineage_store, "_build_s3_client", lambda: _FakeClient())

    assert lineage_store._list_all_lineage_files() == ["a", "b"]


def test_list_all_lineage_files_logs_and_returns_empty_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    test_list_all_lineage_files_logs_and_returns_empty_on_failure: Function description.
    :param monkeypatch:
    :param caplog:
    :returns:
    """

    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)
    monkeypatch.setenv("MODEL_RESULTS_BUCKET", "bucket")

    class ExplodingPaginator:
        """
        ExplodingPaginator: Class description.
        """

        def paginate(self, **kwargs: Any):  # type: ignore[no-untyped-def]
            """
            paginate: Function description.
            :param **kwargs:
            :returns:
            """

            raise RuntimeError("boom")

    class Client:
        """
        Client: Class description.
        """

        def get_paginator(self, name: str) -> ExplodingPaginator:
            """
            get_paginator: Function description.
            :param name:
            :returns:
            """

            return ExplodingPaginator()

    monkeypatch.setattr(lineage_store, "_build_s3_client", lambda: Client())

    with caplog.at_level("ERROR"):
        assert lineage_store._list_all_lineage_files() == []
    assert "Failed to list lineage files" in caplog.text


def test_load_complete_lineage_family_merges_related_graphs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_load_complete_lineage_family_merges_related_graphs: Function description.
    :param tmp_path:
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("AWS_SAM_LOCAL", "1")
    monkeypatch.setattr(lineage_store, "_LOCAL_LINEAGE_DIR", tmp_path)

    graph_a = ArtifactLineageGraph(
        nodes=[
            ArtifactLineageNode(
                artifact_id="a",
                name="ModelA",
                source="primary",
            ),
            ArtifactLineageNode(
                artifact_id="parent-1",
                name="Base",
                source="card",
            ),
        ],
        edges=[
            ArtifactLineageEdge(
                from_node_artifact_id="a",
                to_node_artifact_id="parent-1",
                relationship="base_model",
            ),
        ],
    )
    graph_b = ArtifactLineageGraph(
        nodes=[
            ArtifactLineageNode(
                artifact_id="b",
                name="ModelB",
                source="primary",
            ),
            ArtifactLineageNode(
                artifact_id="a",
                name="ModelA",
                source="card",
            ),
            ArtifactLineageNode(
                artifact_id="extra-1",
                name="Extra",
                source="card",
            ),
            ArtifactLineageNode(
                artifact_id="unnamed-1",
                name=None,
                source="card",
            ),
        ],
        edges=[
            ArtifactLineageEdge(
                from_node_artifact_id="b",
                to_node_artifact_id="a",
                relationship="base_model",
            ),
            ArtifactLineageEdge(
                from_node_artifact_id="b",
                to_node_artifact_id="extra-1",
                relationship="uses",
            ),
        ],
    )
    graph_base = ArtifactLineageGraph(
        nodes=[
            ArtifactLineageNode(
                artifact_id="base-real",
                name="Base",
                source="primary",
            ),
        ],
        edges=[],
    )
    graph_transitive = ArtifactLineageGraph(
        nodes=[
            ArtifactLineageNode(
                artifact_id="d",
                name="ModelD",
                source="primary",
            ),
            ArtifactLineageNode(
                artifact_id="extra-1",
                name="Extra",
                source="card",
            ),
        ],
        edges=[
            ArtifactLineageEdge(
                from_node_artifact_id="d",
                to_node_artifact_id="extra-1",
                relationship="uses",
            )
        ],
    )

    lineage_store.store_lineage("a", graph_a)
    lineage_store.store_lineage("b", graph_b)
    lineage_store.store_lineage("base-real", graph_base)
    lineage_store.store_lineage("d", graph_transitive)

    merged = lineage_store.load_complete_lineage_family("a")

    assert merged is not None
    node_ids = {node.artifact_id for node in merged.nodes}
    assert {"a", "b", "base-real", "d", "extra-1", "unnamed-1"} <= node_ids

    edges = {
        (edge.from_node_artifact_id, edge.to_node_artifact_id, edge.relationship)
        for edge in merged.edges
    }
    # Base was initially a synthetic card node ("parent-1") but is replaced by
    # the primary "base-real" node, so edges must be remapped.
    assert ("a", "base-real", "base_model") in edges
    assert ("a", "parent-1", "base_model") not in edges
