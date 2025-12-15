"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Tests for test cli app module.
"""


from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.CLIApp import CLIApp, build_arg_parser, main
from src.metrics.registry import default_metrics


def _parse_ndjson(text: str) -> list[dict[str, Any]]:
    """
    _parse_ndjson: Function description.
    :param text:
    :returns:
    """

    lines = [line for line in text.strip().splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def test_cli_app_run_outputs_expected_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    test_cli_app_run_outputs_expected_json: Function description.
    :param tmp_path:
    :param capsys:
    :returns:
    """

    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        (
            "https://git.example/repo,"
            "https://hf.co/datasets/sample,"
            "https://hf.co/model\n"
        ),
        encoding="utf-8",
    )
    app = CLIApp(url_file)
    exit_code = app.run()
    results = _parse_ndjson(capsys.readouterr().out)

    assert exit_code == 0
    assert len(results) == 1
    record = results[0]
    assert record["name"] == "model"
    assert record["category"] == "MODEL"

    for metric in default_metrics():
        key = metric.key
        assert key in record
        assert f"{key}_latency" in record

    size_score = record["size_score"]
    assert isinstance(size_score, dict)
    assert set(size_score) == {
        "raspberry_pi",
        "jetson_nano",
        "desktop_pc",
        "aws_server",
    }


def test_cli_main_calls_cli_app(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    test_cli_main_calls_cli_app: Function description.
    :param tmp_path:
    :param capsys:
    :returns:
    """

    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        ",,https://hf.co/model\n",
        encoding="utf-8",
    )
    exit_code = main([str(url_file)])
    results = _parse_ndjson(capsys.readouterr().out)

    assert exit_code == 0
    assert len(results) == 1
    assert results[0]["name"] == "model"
    assert results[0]["category"] == "MODEL"


def test_build_arg_parser_requires_url_file() -> None:
    """
    test_build_arg_parser_requires_url_file: Function description.
    :param:
    :returns:
    """

    parser = build_arg_parser()
    namespace = parser.parse_args(["/tmp/input.txt"])
    assert Path(namespace.url_file) == Path("/tmp/input.txt")
