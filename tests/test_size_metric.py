from __future__ import annotations

from typing import Iterable, Tuple

import pytest

from src.metrics.size import SizeMetric


class _FakeHFClient:
    def __init__(self, files: Iterable[Tuple[str, int]]) -> None:
        self._files = list(files)
        self.calls: list[str] = []

    def list_model_files(
        self, repo_id: str, *, recursive: bool = True
    ) -> list[tuple[str, int]]:
        self.calls.append(repo_id)
        return list(self._files)


def _gb(gb: float) -> int:
    return int(gb * (1024 ** 3))


def test_size_metric_chooses_best_variant() -> None:
    files = [
        ("pytorch_model-00001-of-00002.bin", _gb(2.0)),
        ("pytorch_model-00002-of-00002.bin", _gb(2.0)),
        ("gguf/Q4_K_M/model-q4_0.gguf", _gb(0.8)),
        ("int4/model.bin", _gb(0.4)),
    ]
    metric = SizeMetric(hf_client=_FakeHFClient(files))

    scores = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    assert set(scores) == {
        "raspberry_pi",
        "jetson_nano",
        "desktop_pc",
        "aws_server",
    }
    # With a 0.4 GB variant, all but raspberry_pi should be 1.0
    assert scores["jetson_nano"] == pytest.approx(1.0)
    assert scores["desktop_pc"] == pytest.approx(1.0)
    assert scores["aws_server"] == pytest.approx(1.0)
    # Raspberry Pi scores within its bin for 0.4 GB
    # ideal=0.25, hard=1.0 => 1 - (0.4-0.25)/0.75 ~= 0.80
    assert scores["raspberry_pi"] == pytest.approx(0.80, rel=1e-2)


def test_size_metric_filters_non_weight_files() -> None:
    files = [
        ("README.md", 10_000),
        ("config.json", 1_000),
        ("tokenizer.json", 5_000),
        ("model.safetensors", _gb(0.2)),
    ]
    metric = SizeMetric(hf_client=_FakeHFClient(files))

    scores = metric.compute({"hf_url": "https://huggingface.co/org/model"})

    # 0.2 GB is under every 'ideal' threshold, so all 1.0
    assert all(value == pytest.approx(1.0) for value in scores.values())


def test_size_metric_no_hf_url_returns_zeros() -> None:
    metric = SizeMetric(hf_client=_FakeHFClient([]))
    scores = metric.compute({})
    assert all(value == pytest.approx(0.0) for value in scores.values())


def test_size_metric_no_weight_files_returns_zeros() -> None:
    files = [("README.md", 1000), ("config.json", 2000)]
    metric = SizeMetric(hf_client=_FakeHFClient(files))

    scores = metric.compute({"hf_url": "https://huggingface.co/org/model"})
    assert all(value == pytest.approx(0.0) for value in scores.values())
