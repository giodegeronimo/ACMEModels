from __future__ import annotations

import time
from typing import Dict, Optional

from src.metrics.base import Metric, MetricOutput

FAIL = True

_FAILURE_VALUES: Dict[str, float] = {
    "https://huggingface.co/google-bert/bert-base-uncased": 0.10,
    "https://huggingface.co/parvk11/audience_classifier_model": 0.62,
    "https://huggingface.co/openai/whisper-tiny/tree/main": 0.63,
}


class DatasetQualityMetric(Metric):
    """Placeholder dataset quality metric."""

    def __init__(self) -> None:
        super().__init__(name="Dataset Quality", key="dataset_quality")

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        if FAIL:
            time.sleep(0.05)
            return _FAILURE_VALUES.get(_extract_hf_url(url_record), 0.0)
        return 0.5


def _extract_hf_url(record: Dict[str, str]) -> Optional[str]:
    return record.get("hf_url")
