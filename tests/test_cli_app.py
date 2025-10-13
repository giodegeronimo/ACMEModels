
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from src.CLIApp import CLIApp, build_arg_parser, main
from src.metrics.registry import default_metrics


def _parse_output(
    text: str,
) -> tuple[list[dict[str, str]], list[list[dict[str, Any]]]]:
    stripped = text.strip()
    decoder = json.JSONDecoder()
    first_obj, idx = decoder.raw_decode(stripped)
    manifest = cast(list[dict[str, str]], first_obj)
    remainder = stripped[idx:].lstrip()
    second_obj, _ = decoder.raw_decode(remainder)
    metrics = cast(list[list[dict[str, Any]]], second_obj)
    return manifest, metrics


def test_cli_app_run_outputs_expected_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
    manifest, metrics = _parse_output(capsys.readouterr().out)

    assert exit_code == 0
    assert manifest == [
        {
            "git_url": "https://git.example/repo",
            "ds_url": "https://hf.co/datasets/sample",
            "hf_url": "https://hf.co/model",
        }
    ]
    assert len(metrics) == 1
    assert len(metrics[0]) == len(default_metrics())
    first_result = metrics[0][0]
    for field in ("metric", "key", "value", "latency_ms", "details", "error"):
        assert field in first_result


def test_cli_main_calls_cli_app(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        ",,https://hf.co/model\n",
        encoding="utf-8",
    )
    exit_code = main([str(url_file)])
    manifest, metrics = _parse_output(capsys.readouterr().out)

    assert exit_code == 0
    assert manifest == [{"hf_url": "https://hf.co/model"}]
    assert len(metrics) == 1


def test_build_arg_parser_requires_url_file() -> None:
    parser = build_arg_parser()
    namespace = parser.parse_args(["/tmp/input.txt"])
    assert Path(namespace.url_file) == Path("/tmp/input.txt")
