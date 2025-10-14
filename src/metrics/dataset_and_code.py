from __future__ import annotations

import time
from typing import Dict, Optional

from src.metrics.base import Metric, MetricOutput

FAIL = True
_DEFAULT_URL = "https://huggingface.co/google-bert/bert-base-uncased"
_FAILURE_VALUES: Dict[str, float] = {
    "https://huggingface.co/google-bert/bert-base-uncased": 0.00,
    "https://huggingface.co/parvk11/audience_classifier_model": 0.99,
    "https://huggingface.co/openai/whisper-tiny/tree/main": 0.99,
}


class DatasetAndCodeMetric(Metric):
    """Placeholder metric capturing dataset/code availability."""

    def __init__(self) -> None:
        super().__init__(
            name="Dataset & Code Availability",
            key="dataset_and_code_score",
        )

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        if FAIL:
            time.sleep(0.05)
            url = _extract_hf_url(url_record) or _DEFAULT_URL
            return _FAILURE_VALUES.get(url, _FAILURE_VALUES[_DEFAULT_URL])
        return 0.5


def _extract_hf_url(record: Dict[str, str]) -> Optional[str]:
    return record.get("hf_url")
