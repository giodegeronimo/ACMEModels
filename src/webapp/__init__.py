"""Flask application factory and shared setup for the ACME Models web UI."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from flask import Flask, current_app

from .data_store import DataStore


def create_app(config: Dict[str, Any] | None = None) -> Flask:
    """Create and configure the Flask application."""

    instance_path = Path(__file__).resolve().parent
    static_folder = instance_path / "static"
    template_folder = instance_path / "templates"

    app = Flask(
        __name__,
        static_folder=str(static_folder),
        template_folder=str(template_folder),
    )
    app.config.setdefault("JSON_SORT_KEYS", False)
    app.config.setdefault("DATA_STORE", DataStore())

    if config:
        app.config.update(config)

    from .api import api_bp
    from .views import ui_bp

    app.register_blueprint(ui_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    @app.context_processor
    def inject_global_context() -> Dict[str, Any]:
        """Provide template-wide variables."""
        return {
            "app_name": "ACME Model Registry",
            "current_year": datetime.now(timezone.utc).year,
        }

    @app.template_filter("datetimeformat")
    def datetimeformat_filter(value: float) -> str:
        """Render a UNIX timestamp as an ISO-8601 string."""
        try:
            timestamp = datetime.fromtimestamp(float(value), timezone.utc)
        except (TypeError, ValueError):
            return "Unknown"
        return timestamp.strftime("%Y-%m-%d %H:%M UTC")

    return app


def get_store(app: Flask | None = None) -> DataStore:
    """Retrieve the shared data store. Accepts an optional app override."""
    ctx_app = app or current_app
    store = ctx_app.config.get("DATA_STORE")
    if not isinstance(store, DataStore):
        raise RuntimeError("DATA_STORE config must be a DataStore instance")
    return store
