"""Server-rendered views for the ACME Models prototype UI."""

from __future__ import annotations

from flask import Blueprint, abort, current_app, render_template

from .data_store import DataStore

ui_bp = Blueprint("ui", __name__)


@ui_bp.route("/")
def dashboard() -> str:
    """Landing page showing high-level metrics."""
    store = _get_store()
    stats = store.stats()
    ingest_requests = store.recent_ingest_requests()
    return render_template(
        "dashboard.html",
        stats=stats,
        ingest_requests=ingest_requests,
    )


@ui_bp.route("/models")
def models_directory() -> str:
    """Interactive model directory."""
    return render_template("models.html")


@ui_bp.route("/models/<path:model_id>")
def model_details(model_id: str) -> str:
    """Detail view for a specific model."""
    store = _get_store()
    model = store.get_model(model_id)
    if model is None:
        abort(404)
    return render_template("model_detail.html", model=model)


@ui_bp.route("/ingest")
def request_ingest() -> str:
    """Form for requesting a model ingestion."""
    return render_template("ingest.html")


@ui_bp.route("/license-check")
def license_check_form() -> str:
    """Form for assessing license compatibility."""
    return render_template("license_check.html")


@ui_bp.route("/admin/reset")
def admin_reset() -> str:
    """Administrative page for restoring defaults."""
    return render_template("reset.html")


def _get_store() -> DataStore:
    store = current_app.config.get("DATA_STORE")
    if not isinstance(store, DataStore):
        raise RuntimeError("DATA_STORE config must be a DataStore instance")
    return store
