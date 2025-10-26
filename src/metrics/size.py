from __future__ import annotations

import logging
import re
import time
from typing import Dict, Iterable, Mapping, Optional, Protocol, Tuple

from src.clients.hf_client import HFClient
from src.metrics.base import Metric
from src.utils.env import fail_stub_active

_LOGGER = logging.getLogger(__name__)

FAIL = True
_DEFAULT_URL = "https://huggingface.co/google-bert/bert-base-uncased"

# Stub mapping used when FAIL is active (deterministic behavior)
_FAILURE_VALUES: Dict[str, Dict[str, float]] = {
    "https://huggingface.co/google-bert/bert-base-uncased": {
        "raspberry_pi": 0.34,
        "jetson_nano": 0.5,
        "desktop_pc": 0.83,
        "aws_server": 0.84,
    },
    "https://huggingface.co/parvk11/audience_classifier_model": {
        "raspberry_pi": 1.0,
        "jetson_nano": 1.0,
        "desktop_pc": 0.99,
        "aws_server": 0.99,
    },
    "https://huggingface.co/openai/whisper-tiny/tree/main": {
        "raspberry_pi": 0.4,
        "jetson_nano": 0.8,
        "desktop_pc": 0.99,
        "aws_server": 0.99,
    },
}

# Weight file identification
_WEIGHT_EXTS = {
    ".safetensors",
    ".bin",
    ".pt",
    ".ckpt",
    ".gguf",
    ".ggml",
    ".h5",
    ".onnx",
    ".tflite",
}
_WEIGHT_BASENAMES = {
    "pytorch_model.bin",
    "pytorch_model-",  # shards
    "model.safetensors",
    "consolidated.",
    "flax_model.",
    "adapter_model.bin",
}

# Variant detection
_VARIANT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("gguf-q4", re.compile(r"gguf.*q4", re.IGNORECASE)),
    ("gguf-q5", re.compile(r"gguf.*q5", re.IGNORECASE)),
    ("gguf-q8", re.compile(r"gguf.*q8", re.IGNORECASE)),
    ("gguf", re.compile(r"gguf", re.IGNORECASE)),
    ("gptq", re.compile(r"gptq", re.IGNORECASE)),
    ("awq", re.compile(r"awq", re.IGNORECASE)),
    ("int4", re.compile(r"(int4|4[-_ ]?bit)", re.IGNORECASE)),
    ("int8", re.compile(r"(int8|8[-_ ]?bit)", re.IGNORECASE)),
    ("bf16", re.compile(r"bf16", re.IGNORECASE)),
    ("fp16", re.compile(r"fp16", re.IGNORECASE)),
    ("fp32", re.compile(r"fp32", re.IGNORECASE)),
    ("onnx", re.compile(r"onnx", re.IGNORECASE)),
    ("tflite", re.compile(r"tflite", re.IGNORECASE)),
)

# Device bins: (ideal_max_gb, hard_max_gb)
_DEVICE_BINS: dict[str, Tuple[float, float]] = {
    "raspberry_pi": (0.25, 1.0),
    "jetson_nano": (0.5, 2.0),
    "desktop_pc": (2.0, 8.0),
    "aws_server": (8.0, 32.0),
}


class _HFClientProtocol(Protocol):
    def list_model_files(
        self, repo_id: str, *, recursive: bool = True
    ) -> list[tuple[str, int]]: ...


class SizeMetric(Metric):
    """Compute per-device compatibility from model artifact sizes.

    - Lists model files on HF and filters to likely weight artifacts
    - Groups files into quantization/format variants
    - Computes total size per variant and maps to device scores
    - Final per-device score picks the best (max) variant
    """

    def __init__(self, hf_client: Optional[_HFClientProtocol] = None) -> None:
        super().__init__(name="Size Score", key="size_score")
        self._hf: _HFClientProtocol = hf_client or HFClient()

    def compute(self, url_record: Dict[str, str]) -> Mapping[str, float]:
        hf_url = _extract_hf_url(url_record)
        if fail_stub_active(FAIL):
            time.sleep(0.05)
            url = hf_url or _DEFAULT_URL
            return _FAILURE_VALUES.get(url, _FAILURE_VALUES[_DEFAULT_URL])
        if not hf_url:
            return _zero_scores()

        try:
            files = self._hf.list_model_files(hf_url)
        except Exception as exc:
            _LOGGER.debug("Unable to list files for %s: %s", hf_url, exc)
            return _zero_scores()

        weight_files = _filter_weight_files(files)
        if not weight_files:
            return _zero_scores()

        variants = _group_by_variant(weight_files)
        if not variants:
            return _zero_scores()

        variant_details = {
            name: total_bytes / (1024.0 ** 3)
            for name, total_bytes in variants.items()
        }

        _LOGGER.info(
            "Size metric: evaluated %d variant(s) for %s -> %s",
            len(variant_details),
            hf_url,
            {k: f"{v:.2f} GB" for k, v in variant_details.items()},
        )

        # For each device, compute max score across variants
        scores: Dict[str, float] = {}
        for device, (ideal, hard) in _DEVICE_BINS.items():
            device_scores: Dict[str, float] = {}
            for variant_name, total_gb in variant_details.items():
                device_scores[variant_name] = _piecewise_linear_score(
                    total_gb,
                    ideal,
                    hard,
                )
            best = max(device_scores.values()) if device_scores else 0.0
            scores[device] = float(best)
            _LOGGER.info(
                "Size metric: device=%s ideal<=%.2fGB hard<=%.2fGB "
                "variant_scores=%s best=%.2f",
                device,
                ideal,
                hard,
                {k: f"{v:.2f}" for k, v in device_scores.items()},
                best,
            )

        return scores


def _zero_scores() -> Dict[str, float]:
    return {
        "raspberry_pi": 0.0,
        "jetson_nano": 0.0,
        "desktop_pc": 0.0,
        "aws_server": 0.0,
    }


def _filter_weight_files(
    files: Iterable[tuple[str, int]]
) -> list[tuple[str, int]]:
    results: list[tuple[str, int]] = []
    for path, size in files:
        lower = path.lower()
        if any(lower.endswith(ext) for ext in _WEIGHT_EXTS):
            results.append((path, size))
            continue
        if any(base in lower for base in _WEIGHT_BASENAMES):
            results.append((path, size))
    return results


def _group_by_variant(files: Iterable[tuple[str, int]]) -> Dict[str, int]:
    buckets: Dict[str, int] = {}
    for path, size in files:
        name = path.lower()
        variant = "default"
        for label, pattern in _VARIANT_PATTERNS:
            if pattern.search(name):
                variant = label
                break
        buckets[variant] = buckets.get(variant, 0) + int(size)
    return buckets


def _piecewise_linear_score(x_gb: float, ideal: float, hard: float) -> float:
    if x_gb <= ideal:
        return 1.0
    if x_gb >= hard:
        return 0.0
    return float(1.0 - (x_gb - ideal) / (hard - ideal))


def _extract_hf_url(record: Dict[str, str]) -> Optional[str]:
    return record.get("hf_url")
