from __future__ import annotations

import re
from typing import Any, Dict, Optional

from src.clients.hf_client import HFClient
from src.metrics.base import Metric, MetricOutput

_DATASET_URL_PATTERN = re.compile(
    r"https?://huggingface\.co/(?:datasets/)?[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
    re.IGNORECASE,
)
_CODE_URL_PATTERN = re.compile(
    r"https?://(?:github\.com|gitlab\.com|bitbucket\.org)/"
    r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
    re.IGNORECASE,
)


class DatasetAndCodeMetric(Metric):
    """Score availability of datasets and example code for a model."""

    def __init__(self, hf_client: Optional[HFClient] = None) -> None:
        super().__init__(
            name="Dataset & Code Availability",
            key="dataset_and_code_score",
        )
        self._hf_client = hf_client or HFClient()

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        dataset_score = 0.5 if _has_explicit_dataset(url_record) else 0.0
        code_score = 0.5 if _has_explicit_code(url_record) else 0.0

        hf_url = _extract_hf_url(url_record)
        readme_text: Optional[str] = None

        if dataset_score == 0.0 and hf_url:
            dataset_score = self._score_dataset_from_hub(hf_url)
            if dataset_score == 0.0:
                readme_text = readme_text or self._safe_readme(hf_url)
                if readme_text and _DATASET_URL_PATTERN.search(readme_text):
                    dataset_score = 0.5

        if code_score == 0.0 and hf_url:
            readme_text = readme_text or self._safe_readme(hf_url)
            if readme_text and _CODE_URL_PATTERN.search(readme_text):
                code_score = 0.5

        return min(dataset_score + code_score, 1.0)

    def _score_dataset_from_hub(self, hf_url: str) -> float:
        model_info = self._safe_model_info(hf_url)
        if not model_info:
            return 0.0

        datasets = getattr(model_info, "datasets", None)
        if datasets:
            return 0.5

        card_data = getattr(model_info, "card_data", None)
        if isinstance(card_data, dict) and card_data.get("datasets"):
            return 0.5

        return 0.0

    def _safe_model_info(self, hf_url: str) -> Any:
        try:
            return self._hf_client.get_model_info(hf_url)
        except Exception:
            return None

    def _safe_readme(self, hf_url: str) -> Optional[str]:
        try:
            readme = self._hf_client.get_model_readme(hf_url)
        except Exception:
            return None
        return readme or None


def _extract_hf_url(record: Dict[str, str]) -> Optional[str]:
    return record.get("hf_url")


def _has_explicit_dataset(record: Dict[str, str]) -> bool:
    ds_url = record.get("ds_url")
    return bool(ds_url and ds_url.strip())


def _has_explicit_code(record: Dict[str, str]) -> bool:
    git_url = record.get("git_url")
    return bool(git_url and git_url.strip())
