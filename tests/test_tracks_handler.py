"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for GET /tracks handler.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from backend.src.handlers.tracks import app as handler


def _event() -> Dict[str, Any]:
    """
    _event: Function description.
    :param:
    :returns:
    """

    return {
        "requestContext": {"http": {"method": "GET", "path": "/tracks"}},
        "headers": {},
    }


def test_tracks_handler_returns_default_tracks(monkeypatch) -> None:
    """
    test_tracks_handler_returns_default_tracks: Function description.
    :param monkeypatch:
    :returns:
    """

    response = handler.lambda_handler(_event(), None)

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body == {
        "plannedTracks": [
            "Performance track",
            "Access control track",
            "High assurance track",
            "Other Security track",
        ]
    }
