from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.CLIApp import CLIApp, build_arg_parser, main


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

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload[0]["git_url"] == "https://git.example/repo"
    assert payload[0]["ds_url"] == "https://hf.co/datasets/sample"
    assert payload[0]["hf_url"] == "https://hf.co/model"


def test_cli_main_calls_cli_app(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(",,https://hf.co/model\n", encoding="utf-8")

    exit_code = main([str(url_file)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == [{"hf_url": "https://hf.co/model"}]


def test_build_arg_parser_requires_url_file() -> None:
    parser = build_arg_parser()
    namespace = parser.parse_args(["/tmp/input.txt"])

    assert Path(namespace.url_file) == Path("/tmp/input.txt")
