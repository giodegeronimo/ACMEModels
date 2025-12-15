"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Unit tests for rating persistence helpers.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from src.storage import ratings_store
from src.storage.ratings_store import RatingStoreError


def test_normalize_rating_payload_fills_defaults_and_coerces_values() -> None:
    """
    test_normalize_rating_payload_fills_defaults_and_coerces_values: Function description.
    :param:
    :returns:
    """

    payload: Dict[str, Any] = {
        "name": " ",
        "category": "",
        "net_score": 2.5,
        "net_score_latency": -3,
        "size_score": {"raspberry_pi": True},
    }

    normalized = ratings_store._normalize_rating_payload(payload)

    assert normalized["name"] == "unknown"
    assert normalized["category"] == "MODEL"
    assert normalized["net_score"] == 1.0
    assert normalized["net_score_latency"] == 0.0
    assert normalized["size_score"]["raspberry_pi"] == 1.0
    assert normalized["size_score"]["aws_server"] == 0.0


def test_coerce_helpers_cover_nan_and_boolean_latency() -> None:
    """
    test_coerce_helpers_cover_nan_and_boolean_latency: Function description.
    :param:
    :returns:
    """

    normalized = ratings_store._normalize_rating_payload(
        {"net_score": float("nan"), "net_score_latency": True}
    )
    assert normalized["net_score"] == 0.0
    assert normalized["net_score_latency"] == 0.0
    assert (
        ratings_store._normalize_rating_payload(
            {"net_score_latency": float("nan")}
        )["net_score_latency"]
        == 0.0
    )


def test_store_and_load_rating_local_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_store_and_load_rating_local_roundtrip: Function description.
    :param tmp_path:
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setenv("AWS_SAM_LOCAL", "1")
    monkeypatch.setattr(ratings_store, "_LOCAL_RATINGS_DIR", tmp_path)

    rating: Dict[str, Any] = {"name": "demo", "net_score": float("nan")}
    ratings_store.store_rating("abc123", rating)

    loaded = ratings_store.load_rating("abc123")

    assert loaded is not None
    assert loaded["name"] == "demo"
    assert loaded["net_score"] == 0.0


def test_store_rating_rejects_non_object_payload() -> None:
    """
    test_store_rating_rejects_non_object_payload: Function description.
    :param:
    :returns:
    """

    with pytest.raises(RatingStoreError):
        ratings_store.store_rating("abc123", [])  # type: ignore[arg-type]


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
        self.head_calls: list[tuple[str, str]] = []
        self.raise_on_get: Optional[Exception] = None

    def head_object(self, *, Bucket: str, Key: str) -> Dict[str, Any]:
        """
        head_object: Function description.
        :param Bucket:
        :param Key:
        :returns:
        """

        self.head_calls.append((Bucket, Key))
        if (Bucket, Key) not in self.objects:
            raise RuntimeError("missing")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

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
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


def test_store_and_load_rating_s3_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_store_and_load_rating_s3_roundtrip: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)
    monkeypatch.setenv("MODEL_RESULTS_BUCKET", "bucket")
    monkeypatch.setenv("MODEL_RESULTS_PREFIX", "ratings")
    client = _FakeS3Client()
    monkeypatch.setattr(ratings_store, "_build_s3_client", lambda: client)

    rating: Dict[str, Any] = {"name": "demo"}
    ratings_store.store_rating("abc123", rating)
    loaded = ratings_store.load_rating("abc123")

    assert loaded is not None
    assert client.put_calls == [("bucket", "ratings/abc123.json")]
    assert loaded["name"] == "demo"
    assert loaded["net_score"] == 0.0

    raw = json.loads(client.objects[("bucket", "ratings/abc123.json")])
    assert raw["name"] == "demo"


def test_load_rating_returns_none_on_s3_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_load_rating_returns_none_on_s3_errors: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)
    monkeypatch.setenv("MODEL_RESULTS_BUCKET", "bucket")
    client = _FakeS3Client()
    client.raise_on_get = RuntimeError("boom")
    monkeypatch.setattr(ratings_store, "_build_s3_client", lambda: client)

    assert ratings_store.load_rating("abc123") is None


def test_build_s3_client_raises_when_boto3_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_build_s3_client_raises_when_boto3_missing: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.setattr(ratings_store, "_S3_CLIENT", None)
    monkeypatch.setattr(ratings_store, "boto3", None)
    with pytest.raises(RatingStoreError, match="boto3 is required for rating storage"):
        ratings_store._build_s3_client()


def test_store_rating_logs_and_raises_on_s3_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    test_store_rating_logs_and_raises_on_s3_failure: Function description.
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

    monkeypatch.setattr(ratings_store, "_build_s3_client", lambda: ExplodingClient())

    rating: Dict[str, Any] = {"name": "demo"}
    with caplog.at_level("ERROR"):
        with pytest.raises(RatingStoreError, match="Failed to store rating"):
            ratings_store.store_rating("abc123", rating)

    assert "Failed to store rating for artifact_id=abc123" in caplog.text


def test_warm_s3_connection_uses_head_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_warm_s3_connection_uses_head_when_enabled: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)
    monkeypatch.setenv("ACME_WARM_S3_ON_INIT", "1")
    monkeypatch.setenv("MODEL_RESULTS_BUCKET", "bucket")
    monkeypatch.setenv("MODEL_RESULTS_PREFIX", "ratings")
    monkeypatch.setenv("MODEL_RESULTS_WARMUP_KEY", "ratings/__warmup__.json")
    client = _FakeS3Client()
    client.objects[("bucket", "ratings/__warmup__.json")] = b"{}"
    monkeypatch.setattr(ratings_store, "_build_s3_client", lambda: client)

    ratings_store._warm_s3_connection()

    assert client.head_calls == [("bucket", "ratings/__warmup__.json")]


def test_build_s3_client_caches_instance_and_passes_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_build_s3_client_caches_instance_and_passes_region: Function description.
    :param monkeypatch:
    :returns:
    """

    calls: list[Dict[str, Any]] = []

    class FakeConfig:
        """
        FakeConfig: Class description.
        """

        def __init__(self, **kwargs: Any) -> None:
            """
            __init__: Function description.
            :param **kwargs:
            :returns:
            """

            self.kwargs = kwargs

    class FakeBoto3:
        @staticmethod
        def client(service: str, **kwargs: Any) -> Any:
            """
            client: Function description.
            :param service:
            :param **kwargs:
            :returns:
            """

            calls.append({"service": service, "kwargs": kwargs})
            return _FakeS3Client()

    monkeypatch.setenv("ARTIFACT_STORAGE_REGION", "us-east-1")
    monkeypatch.setattr(ratings_store, "_S3_CLIENT", None)
    monkeypatch.setattr(ratings_store, "Config", FakeConfig)
    monkeypatch.setattr(ratings_store, "boto3", FakeBoto3)

    client1 = ratings_store._build_s3_client()
    client2 = ratings_store._build_s3_client()

    assert client1 is client2
    assert len(calls) == 1
    assert calls[0]["service"] == "s3"
    assert calls[0]["kwargs"]["region_name"] == "us-east-1"
    assert "config" in calls[0]["kwargs"]


def test_warm_s3_connection_computes_default_warmup_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_warm_s3_connection_computes_default_warmup_key: Function description.
    :param monkeypatch:
    :returns:
    """

    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)
    monkeypatch.setenv("ACME_WARM_S3_ON_INIT", "1")
    monkeypatch.setenv("MODEL_RESULTS_BUCKET", "bucket")
    monkeypatch.setenv("MODEL_RESULTS_PREFIX", "ratings")
    monkeypatch.delenv("MODEL_RESULTS_WARMUP_KEY", raising=False)
    client = _FakeS3Client()
    monkeypatch.setattr(ratings_store, "_build_s3_client", lambda: client)

    ratings_store._warm_s3_connection()

    assert client.head_calls == [("bucket", "ratings/__warmup__.json")]


@pytest.mark.parametrize(
    "env",
    [
        {"AWS_SAM_LOCAL": "1", "ACME_WARM_S3_ON_INIT": "1", "MODEL_RESULTS_BUCKET": "bucket"},
        {"AWS_SAM_LOCAL": "", "ACME_WARM_S3_ON_INIT": "0", "MODEL_RESULTS_BUCKET": "bucket"},
        {"AWS_SAM_LOCAL": "", "ACME_WARM_S3_ON_INIT": "1", "MODEL_RESULTS_BUCKET": ""},
    ],
)
def test_warm_s3_connection_skips_when_not_applicable(
    env: Dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_warm_s3_connection_skips_when_not_applicable: Function description.
    :param env:
    :param monkeypatch:
    :returns:
    """

    for key in ("AWS_SAM_LOCAL", "ACME_WARM_S3_ON_INIT", "MODEL_RESULTS_BUCKET"):
        if key in env and env[key]:
            monkeypatch.setenv(key, env[key])
        else:
            monkeypatch.delenv(key, raising=False)
    client = _FakeS3Client()
    monkeypatch.setattr(ratings_store, "_build_s3_client", lambda: client)

    ratings_store._warm_s3_connection()

    assert client.head_calls == []
