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


def _json_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _as_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "t", "yes", "on"}


def _reset_registry_response():
    store = _get_store()
    store.reset()
    return jsonify({"status": "reset"}), 200

@api_bp.get("/health")
def healthcheck() -> tuple[str, int]:
    """Simple readiness probe used by tests or deployment."""
    return jsonify({"status": "ok"}), 200


@api_bp.get("/health/components")
def health_components() -> tuple[str, int]:
    """Return component level diagnostics as described in the OpenAPI spec."""
    store = _get_store()

    window_raw = request.args.get("windowMinutes")
    include_timeline = _as_bool(request.args.get("includeTimeline"))

    if window_raw is None:
        window_minutes = 60
    else:
        try:
            window_minutes = int(window_raw)
        except ValueError:
            return _json_error("windowMinutes must be an integer.", 400)

    try:
        payload = store.component_health(
            window_minutes=window_minutes, include_timeline=include_timeline
        )
    except ValueError as exc:
        return _json_error(str(exc), 400)

    return jsonify(payload), 200


@api_bp.get("/stats")
def stats() -> tuple[str, int]:
    """Return aggregate statistics for dashboards."""
    store = _get_store()
    return jsonify(store.stats()), 200


@api_bp.post("/artifacts")
def enumerate_artifacts():
    """Enumerate artifacts according to the Phase 2 specification."""
    store = _get_store()
    queries = request.get_json(silent=True)
    if not isinstance(queries, list) or not queries:
        return _json_error("Request body must be a non-empty array of artifact queries.", 400)

    offset_raw = request.args.get("offset")
    limit_raw = request.args.get("limit")

    try:
        offset = int(offset_raw) if offset_raw is not None else 0
    except ValueError:
        return _json_error("offset must be an integer.", 400)

    try:
        limit = int(limit_raw) if limit_raw is not None else 50
    except ValueError:
        return _json_error("limit must be an integer.", 400)

    try:
        items, next_offset, total = store.list_artifacts(
            queries, offset=offset, limit=limit
        )
    except NotImplementedError as exc:
        return _json_error(str(exc), 501)
    except ValueError as exc:
        return _json_error(str(exc), 400)

    response = jsonify(items)
    if next_offset is not None:
        response.headers["offset"] = str(next_offset)
    response.headers["X-Total-Count"] = str(total)
    return response, 200


@api_bp.get("/models")
def list_models() -> tuple[str, int]:
    """Legacy model listing endpoint retained for backwards compatibility."""
    store = _get_store()

    limit_raw = request.args.get("limit", default="50")
    cursor = request.args.get("cursor")
    pattern = request.args.get("q")

    try:
        limit = int(limit_raw)
    except ValueError:
        return _json_error("limit must be an integer", 400)

    try:
        items, next_cursor, total = store.list_models(
            limit=limit, cursor=cursor, pattern=pattern
        )
    except ValueError as exc:
        return _json_error(str(exc), 400)

    return jsonify(
        {
            "items": items,
            "next_cursor": next_cursor,
            "total": total,
        }
    ), 200


@api_bp.get("/models/<path:model_id>")
def get_model(model_id: str) -> tuple[str, int]:
    """Legacy model retrieval endpoint retained for backwards compatibility."""
    store = _get_store()
    payload = store.get_model(model_id)
    if not payload:
        return _json_error("model not found", 404)
    return jsonify(payload), 200


@api_bp.post("/models/ingest")
def ingest_model() -> tuple[str, int]:
    """Legacy ingestion endpoint retained for backwards compatibility."""
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
    return _reset_registry_response()


@api_bp.delete("/reset")
def reset_registry() -> tuple[str, int]:
    """Reset the prototype datastore (DELETE variant for spec compliance)."""
    return _reset_registry_response()


@api_bp.get("/ingest/requests")
def ingest_requests() -> tuple[str, int]:
    """Return the latest ingestion requests."""
    store = _get_store()
    payload = store.recent_ingest_requests()
    return jsonify({"items": payload}), 200


@api_bp.get("/artifacts/<string:artifact_type>/<path:artifact_id>")
def artifact_retrieve(artifact_type: str, artifact_id: str):
    store = _get_store()
    try:
        payload = store.get_artifact(artifact_type, artifact_id)
    except NotImplementedError as exc:
        return _json_error(str(exc), 501)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    if payload is None:
        return _json_error("Artifact does not exist.", 404)
    return jsonify(payload), 200


@api_bp.put("/artifacts/<string:artifact_type>/<path:artifact_id>")
def artifact_update(artifact_type: str, artifact_id: str):
    store = _get_store()
    body = request.get_json(silent=True) or {}
    try:
        payload = store.update_artifact(artifact_type, artifact_id, body)
    except NotImplementedError as exc:
        return _json_error(str(exc), 501)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    if payload is None:
        return _json_error("Artifact does not exist.", 404)
    return jsonify(payload), 200


@api_bp.delete("/artifacts/<string:artifact_type>/<path:artifact_id>")
def artifact_delete(artifact_type: str, artifact_id: str):
    store = _get_store()
    try:
        deleted = store.delete_artifact(artifact_type, artifact_id)
    except NotImplementedError as exc:
        return _json_error(str(exc), 501)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    if not deleted:
        return _json_error("Artifact does not exist.", 404)
    return jsonify({"status": "deleted"}), 200


@api_bp.post("/artifact/<string:artifact_type>")
def artifact_create(artifact_type: str):
    store = _get_store()
    body = request.get_json(silent=True) or {}
    try:
        payload = store.create_artifact(artifact_type, body)
    except FileExistsError:
        return _json_error("Artifact exists already.", 409)
    except NotImplementedError as exc:
        return _json_error(str(exc), 501)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    return jsonify(payload), 201


@api_bp.get("/artifact/model/<path:artifact_id>/rate")
def artifact_rate(artifact_id: str):
    store = _get_store()
    payload = store.artifact_rating(artifact_id)
    if payload is None:
        return _json_error("Artifact does not exist.", 404)
    return jsonify(payload), 200


@api_bp.get("/artifact/<string:artifact_type>/<path:artifact_id>/cost")
def artifact_cost(artifact_type: str, artifact_id: str):
    store = _get_store()
    include_dependencies = _as_bool(request.args.get("dependency"))
    try:
        payload = store.artifact_cost(artifact_type, artifact_id, include_dependencies)
    except NotImplementedError as exc:
        return _json_error(str(exc), 501)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    if payload is None:
        return _json_error("Artifact does not exist.", 404)
    return jsonify(payload), 200


@api_bp.put("/authenticate")
def authenticate():
    return _json_error("Authentication is not implemented in this prototype.", 501)


@api_bp.get("/artifact/byName/<string:name>")
def artifact_by_name(name: str):
    store = _get_store()
    try:
        matches = store.search_by_name(name)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    if not matches:
        return _json_error("No such artifact.", 404)
    return jsonify(matches), 200


@api_bp.get("/artifact/<string:artifact_type>/<path:artifact_id>/audit")
def artifact_audit(artifact_type: str, artifact_id: str):
    store = _get_store()
    try:
        entries = store.artifact_audit(artifact_type, artifact_id)
    except NotImplementedError as exc:
        return _json_error(str(exc), 501)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    if entries is None:
        return _json_error("Artifact does not exist.", 404)
    return jsonify(entries), 200


@api_bp.get("/artifact/model/<path:artifact_id>/lineage")
def artifact_lineage(artifact_id: str):
    store = _get_store()
    payload = store.artifact_lineage_graph(artifact_id)
    if payload is None:
        return _json_error("Artifact does not exist.", 404)
    return jsonify(payload), 200


@api_bp.post("/artifact/model/<path:artifact_id>/license-check")
def artifact_license_check(artifact_id: str):
    store = _get_store()
    body = request.get_json(silent=True) or {}
    github_url = body.get("github_url")
    if not github_url:
        return _json_error("github_url is required.", 400)
    try:
        result = store.simple_license_check(artifact_id, github_url)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    if result is None:
        return _json_error("Artifact does not exist.", 404)
    return jsonify(result), 200


@api_bp.post("/artifact/byRegEx")
def artifact_by_regex():
    store = _get_store()
    body = request.get_json(silent=True) or {}
    regex = body.get("regex")
    try:
        matches = store.search_by_regex(regex)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    if not matches:
        return _json_error("No artifact found under this regex.", 404)
    return jsonify(matches), 200


@api_bp.get("/tracks")
def planned_tracks():
    store = _get_store()
    return jsonify(store.planned_tracks()), 200
