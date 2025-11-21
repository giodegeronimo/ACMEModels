import json
import logging
import os
from datetime import datetime, timezone


def _configure_logging() -> logging.Logger:
    level_raw = os.getenv("LOG_LEVEL")
    logger = logging.getLogger(__name__)
    if level_raw and not logger.handlers:
        try:
            numeric_level = int(level_raw)
        except ValueError:
            numeric_level = 0
        logging.basicConfig(
            level=logging.DEBUG if numeric_level >= 2 else logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    return logger


_LOGGER = _configure_logging()


def _log_request(event) -> None:
    if not isinstance(event, dict):
        return
    ctx = (event.get("requestContext") or {}).get("http") or {}
    method = ctx.get("method") or event.get("httpMethod")
    path = ctx.get("path") or event.get("path")
    if not (method or path):
        return
    body = event.get("body")
    if event.get("isBase64Encoded") and isinstance(body, str):
        body_repr = "<base64>"
    elif isinstance(body, str):
        body_repr = body[:256]
    else:
        body_repr = "<missing>" if body is None else str(body)[:256]
    _LOGGER.info(
        "HTTP request method=%s path=%s body=%s",
        method,
        path,
        body_repr,
    )


def lambda_handler(event, context):
    """Return a simple heartbeat payload so load balancers can probe us."""
    _log_request(event)
    payload = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "acme-model-registry",
    }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }
