"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Table-driven checks for auth-required vs open routes.
"""

from __future__ import annotations

import pytest

from backend.src.handlers.artifact_cost import app as cost_handler
from backend.src.handlers.artifact_create import app as create_handler
from backend.src.handlers.artifact_delete import app as delete_handler
from backend.src.handlers.artifact_download import app as download_handler
from backend.src.handlers.artifact_get import app as get_handler
from backend.src.handlers.artifact_license_check import app as license_handler
from backend.src.handlers.artifact_list import app as list_handler
from backend.src.handlers.artifact_rate import app as rate_handler
from backend.src.handlers.artifact_regex import app as regex_handler
from backend.src.handlers.artifact_update import app as update_handler
from backend.src.handlers.health import app as health_handler
from backend.src.handlers.reset import app as reset_handler
from backend.src.handlers.tracks import app as tracks_handler


@pytest.mark.parametrize(
    "handler",
    [
        create_handler.lambda_handler,
        get_handler.lambda_handler,
        update_handler.lambda_handler,
        delete_handler.lambda_handler,
        list_handler.lambda_handler,
        regex_handler.lambda_handler,
        download_handler.lambda_handler,
        rate_handler.lambda_handler,
        cost_handler.lambda_handler,
        license_handler.lambda_handler,
        reset_handler.lambda_handler,
    ],
)
def test_protected_routes_require_auth(handler) -> None:
    """
    test_protected_routes_require_auth: Function description.
    :param handler:
    :returns:
    """

    response = handler({"headers": {}}, {})
    assert response["statusCode"] == 403


@pytest.mark.parametrize(
    "handler,event",
    [
        (health_handler.lambda_handler, {}),
    ],
)
def test_open_routes_allow_anonymous(handler, event) -> None:
    """
    test_open_routes_allow_anonymous: Function description.
    :param handler:
    :param event:
    :returns:
    """

    response = handler(event, {})
    assert response["statusCode"] == 200


def test_tracks_allows_anonymous() -> None:
    """
    test_tracks_allows_anonymous: Function description.
    :param:
    :returns:
    """

    response = tracks_handler.lambda_handler({"headers": {}}, {})
    assert response["statusCode"] == 200
