from __future__ import annotations

import time
from typing import Dict, Mapping, Optional

from src.metrics.base import Metric

FAIL = True
_DEFAULT_URL = "https://huggingface.co/google-bert/bert-base-uncased"

_FAILURE_VALUES: Dict[str, Dict[str, float]] = {
    "https://huggingface.co/google-bert/bert-base-uncased": {
        "raspberry_pi": 0.81,
        "jetson_nano": 0.82,
        "desktop_pc": 0.83,
        "aws_server": 0.84,
    },
    "https://huggingface.co/parvk11/audience_classifier_model": {
        "raspberry_pi": 0.99,
        "jetson_nano": 0.99,
        "desktop_pc": 0.99,
        "aws_server": 0.99,
    },
    "https://huggingface.co/openai/whisper-tiny/tree/main": {
        "raspberry_pi": 0.99,
        "jetson_nano": 0.99,
        "desktop_pc": 0.99,
        "aws_server": 0.99,
    },
}


class SizeMetric(Metric):
    """Placeholder size compatibility metric."""

    def __init__(self) -> None:
        super().__init__(name="Size Score", key="size_score")

    def compute(self, url_record: Dict[str, str]) -> Mapping[str, float]:
        if FAIL:
            time.sleep(0.05)
            url = _extract_hf_url(url_record) or _DEFAULT_URL
            return _FAILURE_VALUES.get(url, _FAILURE_VALUES[_DEFAULT_URL])
        return {
            "raspberry_pi": 0.5,
            "jetson_nano": 0.5,
            "desktop_pc": 0.5,
            "aws_server": 0.5,
        }


def _extract_hf_url(record: Dict[str, str]) -> Optional[str]:
    return record.get("hf_url")
