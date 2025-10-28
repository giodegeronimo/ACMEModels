"""Tests for test parser module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pytest

from src.parser import Parser

FULL_GIT = "https://github.com/acme/models"
FULL_DS = "https://huggingface.co/datasets/acme/sample"
FULL_HF = "https://huggingface.co/acme/sample-model"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            f"{FULL_GIT},{FULL_DS},{FULL_HF}",
            {"git_url": FULL_GIT, "ds_url": FULL_DS, "hf_url": FULL_HF},
        ),
        (
            f"{FULL_GIT}, {FULL_DS}, {FULL_HF}",
            {"git_url": FULL_GIT, "ds_url": FULL_DS, "hf_url": FULL_HF},
        ),
        (
            f"{FULL_GIT},,{FULL_HF}",
            {"git_url": FULL_GIT, "hf_url": FULL_HF},
        ),
        (
            f",{FULL_DS},{FULL_HF}",
            {"ds_url": FULL_DS, "hf_url": FULL_HF},
        ),
        (
            f"{FULL_GIT},{FULL_DS},",
            {"git_url": FULL_GIT, "ds_url": FULL_DS},
        ),
        (
            f"{FULL_GIT},,",
            {"git_url": FULL_GIT},
        ),
        (
            f",{FULL_DS},",
            {"ds_url": FULL_DS},
        ),
        (
            f",,{FULL_HF}",
            {"hf_url": FULL_HF},
        ),
        (
            f" {FULL_GIT} , , ",
            {"git_url": FULL_GIT},
        ),
        (
            " , , ",
            None,
        ),
        (
            "",
            None,
        ),
        (
            "   ",
            None,
        ),
    ],
)
def test_parse_handles_variations(
    tmp_path: Path,
    line: str,
    expected: Optional[Dict[str, str]],
) -> None:
    """Ensure each manifest variation parses to the expected dictionary."""
    url_file = tmp_path / "urls.txt"
    url_file.write_text(f"{line}\n", encoding="utf-8")

    parser = Parser(url_file)
    records = parser.parse()

    if expected:
        assert records == [expected]
    else:
        assert records == []


def test_parse_trims_extra_columns(tmp_path: Path) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        f"{FULL_GIT},{FULL_DS},{FULL_HF},extra\n",
        encoding="utf-8",
    )

    parser = Parser(url_file)
    records = parser.parse()

    assert records == [
        {"git_url": FULL_GIT, "ds_url": FULL_DS, "hf_url": FULL_HF}
    ]


def test_parse_pads_missing_columns(tmp_path: Path) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(f"{FULL_GIT}\n", encoding="utf-8")

    parser = Parser(url_file)
    records = parser.parse()

    assert records == [{"git_url": FULL_GIT}]


def test_parse_collects_multiple_lines(tmp_path: Path) -> None:
    content = "\n".join(
        [
            f"{FULL_GIT},{FULL_DS},{FULL_HF}",
            f",,{FULL_HF}",
            "",
            " , , ",
            f"{FULL_GIT},,{FULL_HF}",
        ]
    )
    url_file = tmp_path / "urls.txt"
    url_file.write_text(content, encoding="utf-8")

    parser = Parser(url_file)
    records = parser.parse()

    assert records == [
        {"git_url": FULL_GIT, "ds_url": FULL_DS, "hf_url": FULL_HF},
        {"hf_url": FULL_HF},
        {"git_url": FULL_GIT, "hf_url": FULL_HF},
    ]


def test_parse_preserves_input_order(tmp_path: Path) -> None:
    url_file = tmp_path / "urls.txt"
    lines = [
        ",,https://huggingface.co/pkg/first",
        f"{FULL_GIT},{FULL_DS},{FULL_HF}",
        ",,https://huggingface.co/pkg/third",
    ]
    url_file.write_text("\n".join(lines), encoding="utf-8")

    parser = Parser(url_file)
    records = parser.parse()

    assert [record["hf_url"] for record in records] == [
        "https://huggingface.co/pkg/first",
        FULL_HF,
        "https://huggingface.co/pkg/third",
    ]


def test_parse_raises_for_missing_file(tmp_path: Path) -> None:
    url_file = tmp_path / "missing.txt"
    parser = Parser(url_file)

    with pytest.raises(FileNotFoundError):
        parser.parse()


def test_parse_ignores_empty_lines(tmp_path: Path) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "\n\n,,https://huggingface.co/pkg/model\n\n",
        encoding="utf-8",
    )

    parser = Parser(url_file)
    records = parser.parse()

    assert records == [{"hf_url": "https://huggingface.co/pkg/model"}]


def test_parse_handles_whitespace_only_entries(tmp_path: Path) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(f"  {FULL_GIT}  ,   ,   \n", encoding="utf-8")

    parser = Parser(url_file)
    records = parser.parse()

    assert records == [{"git_url": FULL_GIT}]


def test_parse_normalizes_fields(tmp_path: Path) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        ",,https://huggingface.co/pkg/model\n",
        encoding="utf-8",
    )

    parser = Parser(url_file)
    records = parser.parse()

    assert "git_url" not in records[0]
    assert "ds_url" not in records[0]
    assert records[0]["hf_url"] == "https://huggingface.co/pkg/model"


def test_parse_outputs_json_serializable(tmp_path: Path) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        f"{FULL_GIT},{FULL_DS},{FULL_HF}\n",
        encoding="utf-8",
    )

    parser = Parser(url_file)
    records = parser.parse()

    serialized = json.dumps(records)
    assert FULL_HF in serialized
