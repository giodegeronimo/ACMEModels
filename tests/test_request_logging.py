from __future__ import annotations

import base64
import logging

from src.utils.request_logging import log_request


def test_log_request_captures_http_event(caplog) -> None:
    logger = logging.getLogger("request_logging_test")
    event = {
        "requestContext": {
            "http": {
                "method": "POST",
                "path": "/artifact/model",
            }
        },
        "queryStringParameters": {"foo": "bar"},
        "pathParameters": {"id": "abc123"},
        "isBase64Encoded": True,
        "body": base64.b64encode(b'{"url":"https://example.com"}').decode(),
    }
    with caplog.at_level(logging.INFO, logger="request_logging_test"):
        log_request(logger, event)
    assert caplog.records
    message = caplog.records[0].message
    assert "artifact/model" in message
    assert "foo" in message
    assert "https://example.com" in message


def test_log_request_skips_non_http_event(caplog) -> None:
    logger = logging.getLogger("request_logging_test_skip")
    with caplog.at_level(logging.INFO, logger="request_logging_test_skip"):
        log_request(logger, {"task": "ingest"})
    assert not caplog.records
