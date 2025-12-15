"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Stub of werkzeug.serving.make_server using wsgiref.
"""

from __future__ import annotations

from wsgiref.simple_server import make_server as _make_server


def make_server(host: str, port: int, app):
    """
    make_server: Function description.
    :param host:
    :param port:
    :param app:
    :returns:
    """

    return _make_server(host, port, app)
