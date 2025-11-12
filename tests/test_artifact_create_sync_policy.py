"""Unit tests for `_can_process_synchronously` heuristics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.src.handlers.artifact_create import app as handler


def test_huggingface_repo_root_forces_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def _fake_head(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("HEAD should not be called for HF repo roots")

    monkeypatch.setattr(handler.requests, "head", _fake_head)

    result = handler._can_process_synchronously(
        "https://huggingface.co/openai/whisper-tiny"
    )

    assert result is False
    assert called is False


def test_huggingface_resolve_allows_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_head(*args, **kwargs):
        return SimpleNamespace(
            headers={"Content-Length": str(1024)},
            raise_for_status=lambda: None,
        )

    monkeypatch.setenv("SYNC_INGEST_MAX_BYTES", str(2 * 1024))
    monkeypatch.setattr(handler.requests, "head", _fake_head)

    result = handler._can_process_synchronously(
        "https://huggingface.co/openai/whisper-tiny/resolve/main/config.json"
    )

    assert result is True
