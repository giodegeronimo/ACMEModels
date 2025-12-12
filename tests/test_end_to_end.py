"""End-to-end and integration tests for the ACME Models CLI pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pytest

from src import runner
from src.CLIApp import CLIApp
from src.metrics.base import Metric, MetricOutput


@dataclass
class _StubMetric(Metric):
    value: MetricOutput

    def __init__(self, name: str, key: str, value: MetricOutput) -> None:
        super().__init__(name=name, key=key)
        self.value = value

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        return self.value


def _install_stub_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    def _factory() -> List[Metric]:
        return [
            _StubMetric("Ramp-up Score", "ramp_up_time", 0.8),
            _StubMetric("Bus Factor", "bus_factor", 0.6),
            _StubMetric(
                "Size Score",
                "size_score",
                {
                    "raspberry_pi": 0.2,
                    "jetson_nano": 0.4,
                    "desktop_pc": 0.9,
                    "aws_server": 1.0,
                },
            ),
            _StubMetric("License", "license", 1.0),
            _StubMetric("Dataset + Code", "dataset_and_code_score", 0.7),
            _StubMetric("Dataset Quality", "dataset_quality", 0.65),
            _StubMetric("Code Quality", "code_quality", 0.75),
            _StubMetric("Performance Claims", "performance_claims", 1.0),
        ]

    monkeypatch.setattr(
        "src.metrics.registry.default_metrics",
        _factory,
    )


def _write_manifest(tmp_path: Path) -> Path:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        ",,https://huggingface.co/org/model-name\n",
        encoding="utf-8",
    )
    return url_file


def _load_records(output: str) -> List[Dict[str, object]]:
    return [json.loads(line) for line in output.strip().splitlines() if line]


def test_cli_app_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_stub_metrics(monkeypatch)
    url_file = _write_manifest(tmp_path)

    app = CLIApp(url_file)
    exit_code = app.run()

    captured = capsys.readouterr()
    records = _load_records(captured.out)

    assert exit_code == 0
    assert len(records) == 1
    record = records[0]
    assert record["name"] == "model-name"
    assert record["category"] == "MODEL"
    assert record["ramp_up_time"] == 0.8
    assert record["net_score"] == pytest.approx(0.79, rel=1e-2)
    size_score = record["size_score"]
    assert isinstance(size_score, dict)
    assert sorted(size_score.keys()) == [
        "aws_server",
        "desktop_pc",
        "jetson_nano",
        "raspberry_pi",
    ]


def test_runner_dispatch_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_stub_metrics(monkeypatch)
    url_file = _write_manifest(tmp_path)

    exit_code = runner.dispatch(["run", str(url_file)])

    captured = capsys.readouterr()
    records = _load_records(captured.out)

    assert exit_code == 0
    assert len(records) == 1
    result = records[0]
    # Ensure latencies are present for representative metrics.
    assert "ramp_up_time_latency" in result
    assert "size_score_latency" in result
