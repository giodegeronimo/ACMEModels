"""

ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

"""
from __future__ import annotations

import os
from typing import Generator

import pytest

from src.webapp import create_app


@pytest.fixture()
def web_app() -> Generator:
    """Provide a configured Flask application for UI tests."""
    app = create_app({"TESTING": True})
    yield app


@pytest.fixture()
def client(web_app):
    """Flask test client fixture."""
    return web_app.test_client()


@pytest.fixture()
def selenium_browser_name() -> str:
    """Read desired Selenium browser from the environment, default to skip."""
    return os.getenv("SELENIUM_BROWSER", "")
