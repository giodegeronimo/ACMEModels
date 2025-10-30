"""REST API blueprint exposing the prototype registry endpoints."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from .data_store import DataStore

api_bp = Blueprint("api", __name__)


def _get_store() -> DataStore:
    store = current_app.config.get("DATA_STORE")
    if not isinstance(store, DataStore):
        raise RuntimeError("DATA_STORE config must be a DataStore instance")
    return store


@api_bp.get("/health")
def healthcheck() -> tuple[str, int]:
    """Simple readiness probe used by tests or deployment."""
    return jsonify({"status": "ok"}), 200


@api_bp.get("/stats")
def stats() -> tuple[str, int]:
    """Return aggregate statistics for dashboards."""
    store = _get_store()
    return jsonify(store.stats()), 200


@api_bp.get("/models")
def list_models() -> tuple[str, int]:
    """Enumerate models with pagination and optional regex filtering."""
    store = _get_store()

    limit_raw = request.args.get("limit", default="50")
    cursor = request.args.get("cursor")
    pattern = request.args.get("q")

    try:
        limit = int(limit_raw)
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400

    try:
        items, next_cursor, total = store.list_models(
            limit=limit, cursor=cursor, pattern=pattern
            )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "items": items,
            "next_cursor": next_cursor,
            "total": total,
        }
    ), 200


@api_bp.get("/models/<path:model_id>")
def get_model(model_id: str) -> tuple[str, int]:
    """Fetch a model by identifier."""
    store = _get_store()
    payload = store.get_model(model_id)
    if not payload:
        return jsonify({"error": "model not found"}), 404
    return jsonify(payload), 200


@api_bp.post("/models/ingest")
def ingest_model() -> tuple[str, int]:
    """Submit a model ingestion request."""
    store = _get_store()
    body = request.get_json(silent=True) or {}

    name = body.get("name")
    source_url = body.get("source_url")
    metrics = body.get("metrics", {})
    submitted_by = body.get("submitted_by", "anonymous")

    if not name or not source_url or not isinstance(metrics, dict):
        return (
            jsonify(
                {
                    "accepted": False,
                    "reason": (
                        "name, source_url, and "
                        "metrics are required fields."
                    ),
                }
            ),
            400,
        )

    result = store.ingest_model(
        name=name,
        source_url=source_url,
        metrics=metrics,
        submitted_by=submitted_by,
    )
    status_code = 201 if result.get("accepted") else 400
    return jsonify(result), status_code


@api_bp.get("/models/<path:model_id>/lineage")
def model_lineage(model_id: str) -> tuple[str, int]:
    """Return a lineage graph for the requested model."""
    store = _get_store()
    payload = store.lineage(model_id)
    if payload is None:
        return jsonify({"error": "model not found"}), 404
    return jsonify(payload), 200


@api_bp.get("/models/<path:model_id>/size-cost")
def model_size_cost(model_id: str) -> tuple[str, int]:
    """Return download size heuristics for a model."""
    store = _get_store()
    payload = store.size_cost(model_id)
    if payload is None:
        return jsonify({"error": "model not found"}), 404
    return jsonify(payload), 200


@api_bp.post("/license-check")
def license_check() -> tuple[str, int]:
    """Assess compatibility between a GitHub repo license and model license."""
    store = _get_store()
    body = request.get_json(silent=True) or {}

    repo_license = body.get("repo_license")
    model_license = body.get("model_license")
    model_id = body.get("model_id")

    if model_id and not model_license:
        model_payload = store.get_model(model_id)
        if model_payload:
            model_license = model_payload.get("license")

    if not repo_license or not model_license:
        return (
            jsonify(
                {
                    "error": (
                        "repo_license and model_license "
                        "(or model_id) are required."
                    ),
                }
            ),
            400,
        )

    result = store.assess_license(repo_license, model_license)
    return jsonify(result), 200


@api_bp.post("/reset")
def reset_state() -> tuple[str, int]:
    """Reset the prototype datastore."""
    store = _get_store()
    store.reset()
    return jsonify({"status": "reset"}), 200


@api_bp.get("/ingest/requests")
def ingest_requests() -> tuple[str, int]:
    """Return the latest ingestion requests."""
    store = _get_store()
    payload = store.recent_ingest_requests()
    return jsonify({"items": payload}), 200
