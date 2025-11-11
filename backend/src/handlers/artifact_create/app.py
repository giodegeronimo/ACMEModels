import json


def lambda_handler(event, context):
    # Just echo the path parameter and body
    artifact_type = event.get("pathParameters", {}).get("artifact_type")
    body = event.get("body")

    try:
        parsed = json.loads(body) if body else {}
    except (json.JSONDecodeError, TypeError):
        parsed = {"error": "body was not valid JSON"}

    return {
        "statusCode": 201,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "message": "Dummy artifact create endpoint working!",
            "artifact_type": artifact_type,
            "received_body": parsed
        })
    }
