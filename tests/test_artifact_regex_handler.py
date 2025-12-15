"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for POST /artifact/byRegEx handler.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import pytest

from backend.src.handlers.artifact_regex import app as handler
from src.models.artifacts import (Artifact, ArtifactData, ArtifactMetadata,
                                  ArtifactType)
from src.storage.errors import ArtifactNotFound
from src.storage.name_index import NameIndexEntry
from src.utils import auth


def _event(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    _event: Function description.
    :param body:
    :returns:
    """

    token = auth.issue_token("tester", is_admin=True)
    return {
        "headers": {"X-Authorization": token},
        "body": json.dumps(body),
    }


class _FakeNameIndexStore:
    """
    _FakeNameIndexStore: Class description.
    """

    def __init__(self, entries: List[NameIndexEntry]) -> None:
        """
        __init__: Function description.
        :param entries:
        :returns:
        """

        self._entries = entries

    def scan(
        self,
        *,
        start_key: Any | None = None,
        limit: int | None = None,
    ) -> Tuple[List[NameIndexEntry], Any | None]:
        """
        scan: Function description.
        :param start_key:
        :param limit:
        :returns:
        """

        return list(self._entries), None


class _FakeMetadataStore:
    """
    _FakeMetadataStore: Class description.
    """

    def __init__(self, records: Dict[str, Artifact]) -> None:
        """
        __init__: Function description.
        :param records:
        :returns:
        """

        self._records = records

    def load(self, artifact_id: str) -> Artifact:
        """
        load: Function description.
        :param artifact_id:
        :returns:
        """

        try:
            return self._records[artifact_id]
        except KeyError as exc:
            raise ArtifactNotFound("missing") from exc


def _artifact(name: str, artifact_id: str) -> Artifact:
    """
    _artifact: Function description.
    :param name:
    :param artifact_id:
    :returns:
    """

    metadata = ArtifactMetadata(
        name=name,
        id=artifact_id,
        type=ArtifactType.MODEL,
    )
    return Artifact(
        metadata=metadata,
        data=ArtifactData(url="https://example.com"),
    )


def test_regex_search_returns_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_regex_search_returns_matches: Function description.
    :param monkeypatch:
    :returns:
    """

    entries = [
        NameIndexEntry("a1", "bert-base-uncased", ArtifactType.MODEL),
        NameIndexEntry("a2", "whisper-tiny", ArtifactType.MODEL),
    ]
    monkeypatch.setattr(handler, "_NAME_INDEX", _FakeNameIndexStore(entries))
    monkeypatch.setattr(
        handler,
        "_METADATA_STORE",
        _FakeMetadataStore(
            {
                entry.artifact_id: _artifact(entry.name, entry.artifact_id)
                for entry in entries
            }
        ),
    )

    response = handler.lambda_handler(_event({"regex": "bert"}), context={})

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert len(body) == 1
    assert body[0]["name"] == "bert-base-uncased"
    assert body[0]["id"] == "a1"


def test_regex_search_handles_invalid_regex() -> None:
    """
    test_regex_search_handles_invalid_regex: Function description.
    :param:
    :returns:
    """

    response = handler.lambda_handler(_event({"regex": "["}), context={})
    assert response["statusCode"] == 400


def test_regex_search_returns_404_when_no_match() -> None:
    """
    test_regex_search_returns_404_when_no_match: Function description.
    :param:
    :returns:
    """

    response = handler.lambda_handler(_event({"regex": "missing"}), context={})
    assert response["statusCode"] == 404


def test_regex_search_falls_back_when_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_regex_search_falls_back_when_metadata_missing: Function description.
    :param monkeypatch:
    :returns:
    """

    entry = NameIndexEntry("missing-id", "lone-model", ArtifactType.MODEL)
    monkeypatch.setattr(handler, "_NAME_INDEX", _FakeNameIndexStore([entry]))
    monkeypatch.setattr(handler, "_METADATA_STORE", _FakeMetadataStore({}))

    response = handler.lambda_handler(_event({"regex": "lone"}), context={})
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body[0]["name"] == "lone-model"
    assert body[0]["id"] == "missing-id"


def test_regex_search_matches_readme(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_regex_search_matches_readme: Function description.
    :param monkeypatch:
    :returns:
    """

    entry = NameIndexEntry(
        "rid",
        "unrelated-name",
        ArtifactType.MODEL,
        readme_excerpt="Supports advanced sentiment analysis",
    )
    monkeypatch.setattr(handler, "_NAME_INDEX", _FakeNameIndexStore([entry]))
    monkeypatch.setattr(
        handler,
        "_METADATA_STORE",
        _FakeMetadataStore(
            {entry.artifact_id: _artifact(entry.name, entry.artifact_id)}
        ),
    )


def test_regex_search_detects_slow_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_regex_search_detects_slow_pattern: Function description.
    :param monkeypatch:
    :returns:
    """

    response = handler.lambda_handler(_event({"regex": "(a+)+$"}), context={})
    assert response["statusCode"] == 400
    entry = NameIndexEntry(
        "rid",
        "unrelated-name",
        ArtifactType.MODEL,
        readme_excerpt="Supports advanced sentiment analysis",
    )
    monkeypatch.setattr(handler, "_NAME_INDEX", _FakeNameIndexStore([entry]))
    monkeypatch.setattr(
        handler,
        "_METADATA_STORE",
        _FakeMetadataStore(
            {entry.artifact_id: _artifact(entry.name, entry.artifact_id)}
        ),
    )
    response = handler.lambda_handler(
        _event({"regex": "sentiment"}),
        context={},
    )
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body[0]["id"] == "rid"
