"""

ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

"""
from __future__ import annotations

import json
from io import BytesIO

from src import webapp


def test_wsgi_interface_handles_query_and_returns_json() -> None:
    """
    test_wsgi_interface_handles_query_and_returns_json: Function description.
    :param:
    :returns:
    """

    app = webapp.create_app()
    captured_status = ""

    def start_response(  # type: ignore[no-untyped-def]
        status: str,
        headers,
        exc_info=None,
    ) -> None:
        """
        start_response: Function description.
        :param status:
        :param headers:
        :param exc_info:
        :returns:
        """

        nonlocal captured_status
        captured_status = status

    environ = {
        "REQUEST_METHOD": "GET",
        "PATH_INFO": "/api/models",
        "QUERY_STRING": "limit=1",
        "CONTENT_LENGTH": "0",
        "wsgi.input": BytesIO(),
    }
    chunks = app(environ, start_response)
    payload = json.loads(b"".join(chunks).decode())

    assert captured_status.startswith("200")
    assert payload["items"]
    assert payload["total"] >= len(payload["items"])


def test_wsgi_interface_reads_post_body() -> None:
    """
    test_wsgi_interface_reads_post_body: Function description.
    :param:
    :returns:
    """

    app = webapp.create_app()

    body = json.dumps({"metrics": {"ramp_up_time": 0.8}}).encode()
    environ = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/api/models/ingest",
        "QUERY_STRING": "",
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": BytesIO(body),
    }

    captured_status = ""

    def start_response(  # type: ignore[no-untyped-def]
        status: str,
        headers,
        exc_info=None,
    ) -> None:
        """
        start_response: Function description.
        :param status:
        :param headers:
        :param exc_info:
        :returns:
        """

        nonlocal captured_status
        captured_status = status

    chunks = app(environ, start_response)
    payload = json.loads(b"".join(chunks).decode())

    assert captured_status.startswith("201")
    assert payload["status"] == "accepted"


def test_header_lookup_is_case_insensitive() -> None:
    """
    test_header_lookup_is_case_insensitive: Function description.
    :param:
    :returns:
    """

    assert webapp._header_lookup([("X-Test", "value")], "x-test") == "value"
    assert webapp._header_lookup([], "x-test") == ""


def test_test_client_covers_remaining_branches() -> None:
    """
    test_test_client_covers_remaining_branches: Function description.
    :param:
    :returns:
    """

    client = webapp.create_app().test_client()

    response = client.get("/api/models/lineage")
    assert response.status_code == 404
    assert response.get_json()["error"] == "not found"

    response = client.post(
        "/api/license-check",
        json={"artifact_id": "x", "github_url": "https://github.com/o/r"},
    )
    assert response.status_code == 200
    assert response.get_json()["compatible"] is True

    text = webapp.Response("hi").get_data(as_text=True)
    assert text == "hi"
