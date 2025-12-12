"""Minimal web app for web tests without external dependencies."""

from __future__ import annotations

import json as json_module
from io import BytesIO
from typing import Any, Dict, Optional

_MODELS = [
    {"id": "acme/solar-safeguard", "name": "Solar Safeguard", "type": "model"},
    {
        "id": "acme/solar-safeguard-edge",
        "name": "Solar Safeguard Edge",
        "type": "model",
    },
]


class Response:
    def __init__(
        self,
        body: str | bytes,
        status: int = 200,
        content_type: str = "",
    ):
        self.status_code = status
        self.content_type = content_type
        self.data = body.encode() if isinstance(body, str) else body

    def get_data(self, as_text: bool = False):
        return self.data.decode() if as_text else self.data

    def get_json(self) -> Dict[str, Any]:
        return json_module.loads(self.data.decode() or "{}")


class WebApp:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    # WSGI interface for make_server
    def __call__(self, environ, start_response):
        method = environ.get("REQUEST_METHOD", "GET").upper()
        raw_path = environ.get("PATH_INFO", "/") or "/"
        query_string = environ.get("QUERY_STRING", "")
        if query_string:
            raw_path = f"{raw_path}?{query_string}"
        length = int(environ.get("CONTENT_LENGTH") or 0)
        body = (
            environ.get("wsgi.input", BytesIO()).read(length)
            if length > 0
            else b""
        )
        response = self._dispatch(raw_path, method, body)
        status_line = f"{response.status_code} OK"
        start_response(
            status_line,
            [("Content-Type", response.content_type or "text/plain")],
        )
        return [response.data]

    def test_client(self):
        return _TestClient(self)

    def _dispatch(
        self,
        path: str,
        method: str,
        body: bytes = b"",
        headers: Optional[Dict[str, str]] = None,
    ) -> Response:
        query = {}
        if "?" in path:
            path, qs = path.split("?", 1)
            query = _parse_query(qs)
        headers = headers or {}

        if path == "/" and method == "GET":
            return _html_response(
                """
                <html>
                <body>
                  <a class="skip-link" href="#main-content">
                    Skip to main content
                  </a>
                  <main id="main-content">
                    <h2>Registry Overview</h2>
                  </main>
                </body>
                </html>
                """
            )

        if path == "/models" and method == "GET":
            rows = "".join(
                f"<tr><td>{m['name']}</td><td>{m['type']}</td></tr>"
                for m in _MODELS
            )
            return _html_response(
                f"""
                <html><body>
                <h1>Model Directory</h1>
                <table class="data-table">{rows}</table>
                </body></html>
                """
            )

        if path.startswith("/models/") and method == "GET":
            model_id = path[len("/models/"):]
            model = next((m for m in _MODELS if m["id"] == model_id), None)
            if model is None:
                return _text_response("Not found", status=404)
            return _html_response(
                f"<html><body><h1>{model['name']}</h1></body></html>"
            )

        if path == "/api/models" and method == "GET":
            limit = int(query.get("limit", "10") or "10")
            q = query.get("q")
            if q and "[" in q:
                return _json_response({"error": "invalid regex"}, status=400)
            items = _MODELS[:limit]
            return _json_response({"items": items, "total": len(_MODELS)})

        if (
            path.startswith("/api/models/")
            and path.endswith("/lineage")
            and method == "GET"
        ):
            parts = path.strip("/").split("/")
            # parts: ["api","models", "<model segments...>", "lineage"]
            if len(parts) >= 4:
                model_id = "/".join(parts[2:-1])
            else:
                model_id = ""
            model = next((m for m in _MODELS if m["id"] == model_id), None)
            if model is None:
                return _json_response({"error": "not found"}, status=404)
            return _json_response(
                {"root": model_id, "nodes": [model_id], "edges": []}
            )

        if path == "/api/models/ingest" and method == "POST":
            payload = json_module.loads(body.decode() or "{}")
            metrics = payload.get("metrics") or {}
            if metrics.get("ramp_up_time", 1.0) < 0.5:
                return _json_response(
                    {"reason": "ramp_up_time below threshold"}, status=400
                )
            return _json_response({"status": "accepted"}, status=201)

        if path == "/api/license-check" and method == "POST":
            payload = json_module.loads(body.decode() or "{}")
            if not payload.get("artifact_id") or not payload.get("github_url"):
                return _json_response(
                    {"error": "artifact_id and github_url required"},
                    status=400,
                )
            return _json_response({"compatible": True})

        return _text_response("Not found", status=404)


class _TestClient:
    def __init__(self, app: WebApp) -> None:
        self.app = app

    def get(self, path: str):
        resp = self.app._dispatch(path, "GET")
        return resp

    def post(self, path: str, json: Optional[Dict[str, Any]] = None):
        payload = json or {}
        body = (json_module.dumps(payload)).encode()
        resp = self.app._dispatch(path, "POST", body=body)
        return resp


def _header_lookup(headers, name: str) -> str:
    for k, v in headers or []:
        if k.lower() == name.lower():
            return v
    return ""


def _parse_query(raw: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for part in raw.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
    return result


def _html_response(body: str, status: int = 200) -> Response:
    return Response(body, status=status, content_type="text/html")


def _json_response(payload: Dict[str, Any], status: int = 200) -> Response:
    return Response(
        json_module.dumps(payload),
        status=status,
        content_type="application/json",
    )


def _text_response(body: str, status: int = 200) -> Response:
    return Response(body, status=status, content_type="text/plain")


def create_app(config: Optional[Dict[str, Any]] = None) -> WebApp:
    return WebApp(config)
