"""

ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

"""
from __future__ import annotations

import pytest


def test_dashboard_renders_accessible(client):
    """
    test_dashboard_renders_accessible: Function description.
    :param client:
    :returns:
    """

    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "<main id=\"main-content\"" in html
    assert "Skip to main content" in html
    assert "Registry Overview" in html


def test_models_view_contains_table(client):
    """
    test_models_view_contains_table: Function description.
    :param client:
    :returns:
    """

    response = client.get("/models")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "<table class=\"data-table\"" in html
    assert "Model Directory" in html


@pytest.mark.parametrize(
    ("path", "status_code"),
    [
        ("/models/acme/solar-safeguard", 200),
        ("/models/not-found-model", 404),
    ],
)
def test_model_detail_status_codes(client, path, status_code):
    """
    test_model_detail_status_codes: Function description.
    :param client:
    :param path:
    :param status_code:
    :returns:
    """

    response = client.get(path)
    assert response.status_code == status_code
