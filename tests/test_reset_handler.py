"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for DELETE /reset handler.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from backend.src.handlers.reset import app as handler
from src.utils import auth


@pytest.fixture
def fake_boto3(monkeypatch: pytest.MonkeyPatch) -> "_FakeBoto3":
    """
    fake_boto3: Function description.
    :param monkeypatch:
    :returns:
    """

    fake = _FakeBoto3()
    monkeypatch.setattr(handler, "boto3", fake)
    return fake


@pytest.fixture(autouse=True)
def _reset_env(
    monkeypatch: pytest.MonkeyPatch,
    fake_boto3: "_FakeBoto3",
) -> None:
    """
    _reset_env: Function description.
    :param monkeypatch:
    :param fake_boto3:
    :returns:
    """

    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)
    monkeypatch.setenv("ARTIFACT_STORAGE_BUCKET", "test-bucket")
    monkeypatch.setenv("ARTIFACT_STORAGE_PREFIX", "artifacts")
    monkeypatch.setenv("ARTIFACT_METADATA_PREFIX", "metadata")
    monkeypatch.setenv("ARTIFACT_NAME_INDEX_TABLE", "name-index")


def _event(headers: Dict[str, str] | None = None) -> Dict[str, Any]:
    """
    _event: Function description.
    :param headers:
    :returns:
    """

    if headers is None:
        headers = {
            "X-Authorization": auth.issue_token("tester", is_admin=True)
        }
    return {"headers": headers}


def test_reset_calls_remote_services(fake_boto3: "_FakeBoto3") -> None:
    """
    test_reset_calls_remote_services: Function description.
    :param fake_boto3:
    :returns:
    """

    response = handler.lambda_handler(
        _event(), {}
    )

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["status"] == "reset"
    assert fake_boto3.s3_client.deleted_batches
    assert fake_boto3.dynamo_resource.table.deleted


def test_reset_local_mode(monkeypatch: pytest.MonkeyPatch,
                          tmp_path: Path) -> None:
    """
    test_reset_local_mode: Function description.
    :param monkeypatch:
    :param tmp_path:
    :returns:
    """

    monkeypatch.setenv("AWS_SAM_LOCAL", "1")
    metadata_dir = tmp_path / "meta"
    artifacts_dir = tmp_path / "blobs"
    metadata_dir.mkdir()
    (metadata_dir / "foo.json").write_text("{}", encoding="utf-8")
    artifacts_dir.mkdir()
    (artifacts_dir / "blob.bin").write_bytes(b"data")
    monkeypatch.setenv("ARTIFACT_METADATA_DIR", str(metadata_dir))
    monkeypatch.setenv("ARTIFACT_STORAGE_DIR", str(artifacts_dir))

    handler.lambda_handler(_event(), {})

    assert not any(metadata_dir.iterdir())
    assert not any(artifacts_dir.iterdir())


class _FakeBoto3:
    """
    _FakeBoto3: Class description.
    """

    def __init__(self) -> None:
        """
        __init__: Function description.
        :param:
        :returns:
        """

        self.s3_client = _FakeS3Client()
        self.dynamo_resource = _FakeDynamoResource()

    def client(self, service: str, **kwargs):
        """
        client: Function description.
        :param service:
        :param **kwargs:
        :returns:
        """

        assert service == "s3"
        return self.s3_client

    def resource(self, service: str, **kwargs):
        """
        resource: Function description.
        :param service:
        :param **kwargs:
        :returns:
        """

        assert service == "dynamodb"
        return self.dynamo_resource


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

        self.deleted_batches: list[list[dict[str, str]]] = []

    def get_paginator(self, name: str):
        """
        get_paginator: Function description.
        :param name:
        :returns:
        """

        assert name == "list_objects_v2"

        class _Paginator:
            """
            _Paginator: Class description.
            """

            def paginate(self, **kwargs):
                """
                paginate: Function description.
                :param **kwargs:
                :returns:
                """

                prefix = kwargs.get("Prefix", "")
                if prefix == "artifacts":
                    return [{"Contents": [{"Key": "artifacts/a"}]}]
                if prefix == "metadata":
                    return [{"Contents": [{"Key": "metadata/a"}]}]
                return []

        return _Paginator()

    def delete_objects(self, Bucket: str, Delete: Dict[str, Any]):
        """
        delete_objects: Function description.
        :param Bucket:
        :param Delete:
        :returns:
        """

        self.deleted_batches.append(Delete["Objects"])


class _FakeDynamoResource:
    """
    _FakeDynamoResource: Class description.
    """

    def __init__(self) -> None:
        """
        __init__: Function description.
        :param:
        :returns:
        """

        self.table = _FakeDynamoTable()

    def Table(self, name: str):
        """
        Table: Function description.
        :param name:
        :returns:
        """

        return self.table


class _FakeDynamoTable:
    """
    _FakeDynamoTable: Class description.
    """

    def __init__(self) -> None:
        """
        __init__: Function description.
        :param:
        :returns:
        """

        self.items = [
            {"normalized_name": "artifact", "artifact_id": "1"},
            {"normalized_name": "model", "artifact_id": "2"},
        ]
        self.deleted: list[dict[str, str]] = []

    def scan(self, **kwargs):
        """
        scan: Function description.
        :param **kwargs:
        :returns:
        """

        chunk = self.items
        self.items = []
        return {"Items": chunk}

    def batch_writer(self):
        """
        batch_writer: Function description.
        :param:
        :returns:
        """

        table = self

        class _Writer:
            """
            _Writer: Class description.
            """

            def __enter__(self_inner):
                """
                __enter__: Function description.
                :param self_inner:
                :returns:
                """

                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                """
                __exit__: Function description.
                :param self_inner:
                :param exc_type:
                :param exc:
                :param tb:
                :returns:
                """

                return False

            def delete_item(self_inner, Key):
                """
                delete_item: Function description.
                :param self_inner:
                :param Key:
                :returns:
                """

                table.deleted.append(Key)

        return _Writer()
