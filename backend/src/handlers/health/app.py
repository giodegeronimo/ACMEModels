import json
import logging
from datetime import datetime, timezone

from src.logging_config import configure_logging

configure_logging()
_LOGGER = logging.getLogger(__name__)


def lambda_handler(event, context):
    """Return a simple heartbeat payload so load balancers can probe us."""
    _LOGGER.debug("Health check received")
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
