"""Stub of werkzeug.serving.make_server using wsgiref."""

from __future__ import annotations

from wsgiref.simple_server import make_server as _make_server


def make_server(host: str, port: int, app):
    return _make_server(host, port, app)
