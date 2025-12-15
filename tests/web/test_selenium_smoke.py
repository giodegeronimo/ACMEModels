"""

ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

"""
from __future__ import annotations


def test_dashboard_heading_visible(client) -> None:
    """
    test_dashboard_heading_visible: Function description.
    :param client:
    :returns:
    """

    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Registry Overview" in html
