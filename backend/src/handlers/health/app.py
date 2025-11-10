import json
from datetime import datetime, timezone


def lambda_handler(event, context):
    """Return a simple heartbeat payload so load balancers can probe us."""
    payload = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "acme-model-registry"
    }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload)
    }
