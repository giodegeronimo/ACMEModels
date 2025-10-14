from __future__ import annotations

import time
from typing import Dict, Optional

from src.metrics.base import Metric, MetricOutput

FAIL = True

_DEFAULT_URL = "https://huggingface.co/google-bert/bert-base-uncased"

_FAILURE_VALUES: Dict[str, float] = {
    "https://huggingface.co/google-bert/bert-base-uncased": 0.41,
    "https://huggingface.co/parvk11/audience_classifier_model": 0.42,
    "https://huggingface.co/openai/whisper-tiny/tree/main": 0.43,
}


class LicenseMetric(Metric):
    """Placeholder license compliance metric."""

    def __init__(self) -> None:
        super().__init__(name="License Score", key="license")

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        if FAIL:
            time.sleep(0.05)
            url = _extract_hf_url(url_record) or _DEFAULT_URL
            return _FAILURE_VALUES.get(url, _FAILURE_VALUES[_DEFAULT_URL])
        return 0.5


def _extract_hf_url(record: Dict[str, str]) -> Optional[str]:
    return record.get("hf_url")
