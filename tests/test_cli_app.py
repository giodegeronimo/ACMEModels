from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.CLIApp import CLIApp, build_arg_parser, main


def test_cli_app_run_outputs_expected_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
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

    stub_git_client = SimpleNamespace(
        get_repo_metadata=lambda url: {"full_name": "example/repo"}
    )
    monkeypatch.setattr("src.CLIApp.GitClient", lambda: stub_git_client)

    app = CLIApp(url_file)
    exit_code = app.run()

    captured = capsys.readouterr()
    first_json, second_json = captured.out.strip().split("\n{\n", 1)
    payload = json.loads(first_json)
    demo = json.loads("{\n" + second_json)

    assert exit_code == 0
    assert payload[0]["git_url"] == "https://git.example/repo"
    assert payload[0]["ds_url"] == "https://hf.co/datasets/sample"
    assert payload[0]["hf_url"] == "https://hf.co/model"
    assert demo == {"demo_git_metadata": {"full_name": "example/repo"}}


def test_cli_main_calls_cli_app(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(",,https://hf.co/model\n", encoding="utf-8")

    monkeypatch.setattr(
        "src.CLIApp.GitClient",
        lambda: SimpleNamespace(get_repo_metadata=lambda _: {}),
    )

    exit_code = main([str(url_file)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == [{"hf_url": "https://hf.co/model"}]


def test_build_arg_parser_requires_url_file() -> None:
    parser = build_arg_parser()
    namespace = parser.parse_args(["/tmp/input.txt"])

    assert Path(namespace.url_file) == Path("/tmp/input.txt")
