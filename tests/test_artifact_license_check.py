from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from src.models import Artifact, ArtifactData, ArtifactMetadata, ArtifactType


class StubGitClient:
    def __init__(self, metadata=None, exc: Exception | None = None) -> None:
        self.metadata = metadata or {}
        self.exc = exc
        self.calls: list[str] = []

    def get_repo_metadata(self, url: str):
        self.calls.append(url)
        if self.exc:
            raise self.exc
        return self.metadata


def _load_handler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module_name = "backend.src.handlers.artifact_license_check.app"
    sys.modules.pop(module_name, None)
    monkeypatch.setenv("AWS_SAM_LOCAL", "1")
    monkeypatch.setenv("ARTIFACT_METADATA_DIR", str(tmp_path / "meta"))
    module = importlib.import_module(module_name)
    module._set_git_client(None)
    return module


def _store_model(handler_module, artifact_id: str, *, artifact_type=ArtifactType.MODEL):
    artifact = Artifact(
        metadata=ArtifactMetadata(
            name="demo-model",
            id=artifact_id,
            type=artifact_type,
        ),
        data=ArtifactData(url="https://example.com/model"),
    )
    handler_module._METADATA_STORE.save(artifact, overwrite=True)


def _event(artifact_id: str, github_url: str | None, headers: dict | None = None):
    event = {
        "pathParameters": {"id": artifact_id},
        "body": json.dumps({"github_url": github_url} if github_url else {}),
        "headers": headers or {},
    }
    return event


def test_license_check_success(tmp_path, monkeypatch):
    handler = _load_handler(tmp_path, monkeypatch)
    _store_model(handler, "abc123")
    stub = StubGitClient(metadata={"license": {"spdx_id": "MIT"}})
    handler._set_git_client(stub)

    resp = handler.lambda_handler(
        _event("abc123", "https://github.com/org/repo", {"X-Authorization": "token"}),
        None,
    )

    assert resp["statusCode"] == 200
    assert json.loads(resp["body"]) is True
    assert stub.calls == ["https://github.com/org/repo"]


def test_non_model_artifact_returns_not_found(tmp_path, monkeypatch):
    handler = _load_handler(tmp_path, monkeypatch)
    _store_model(handler, "abc123", artifact_type=ArtifactType.DATASET)
    stub = StubGitClient(metadata={"license": {"spdx_id": "MIT"}})
    handler._set_git_client(stub)

    resp = handler.lambda_handler(
        _event("abc123", "https://github.com/org/repo", {"X-Authorization": "token"}),
        None,
    )

    assert resp["statusCode"] == 404


def test_invalid_body_returns_bad_request(tmp_path, monkeypatch):
    handler = _load_handler(tmp_path, monkeypatch)
    _store_model(handler, "abc123")
    handler._set_git_client(StubGitClient())

    resp = handler.lambda_handler(
        _event("abc123", None, {"X-Authorization": "token"}),
        None,
    )

    assert resp["statusCode"] == 400


def test_missing_auth_returns_forbidden(tmp_path, monkeypatch):
    handler = _load_handler(tmp_path, monkeypatch)
    _store_model(handler, "abc123")
    handler._set_git_client(StubGitClient(metadata={"license": {"spdx_id": "MIT"}}))

    resp = handler.lambda_handler(
        _event("abc123", "https://github.com/org/repo", {}),
        None,
    )

    assert resp["statusCode"] == 403


def test_repo_not_found(tmp_path, monkeypatch):
    handler = _load_handler(tmp_path, monkeypatch)
    _store_model(handler, "abc123")
    handler._set_git_client(
        StubGitClient(exc=RuntimeError("Failed to retrieve repo metadata: 404"))
    )

    resp = handler.lambda_handler(
        _event("abc123", "https://github.com/org/repo", {"X-Authorization": "token"}),
        None,
    )

    assert resp["statusCode"] == 404


def test_external_failure_returns_bad_gateway(tmp_path, monkeypatch):
    handler = _load_handler(tmp_path, monkeypatch)
    _store_model(handler, "abc123")
    handler._set_git_client(StubGitClient(exc=RuntimeError("timeout")))

    resp = handler.lambda_handler(
        _event("abc123", "https://github.com/org/repo", {"X-Authorization": "token"}),
        None,
    )

    assert resp["statusCode"] == 502
