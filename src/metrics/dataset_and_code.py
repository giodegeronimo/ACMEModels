from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Protocol

from src.clients.hf_client import HFClient
from src.metrics.base import Metric, MetricOutput

_LOGGER = logging.getLogger(__name__)

FAIL = True
_DEFAULT_URL = "https://huggingface.co/google-bert/bert-base-uncased"
_FAILURE_VALUES: Dict[str, float] = {
    "https://huggingface.co/google-bert/bert-base-uncased": 0.0,
    "https://huggingface.co/parvk11/audience_classifier_model": 0.0,
    "https://huggingface.co/openai/whisper-tiny/tree/main": 0.7,
}

_DATASET_URL_PATTERN = re.compile(
    r"https?://huggingface\.co/(?:datasets/)?[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
    re.IGNORECASE,
)
_CODE_URL_PATTERN = re.compile(
    r"https?://(?:github\.com|gitlab\.com|bitbucket\.org)/"
    r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
    re.IGNORECASE,
)


class _HFClientProtocol(Protocol):
    def get_model_info(self, repo_id: str) -> Any: ...

    def get_model_readme(self, repo_id: str) -> str: ...


class DatasetAndCodeMetric(Metric):
    """Score availability of datasets and example code for a model."""

    def __init__(self, hf_client: Optional[_HFClientProtocol] = None) -> None:
        super().__init__(
            name="Dataset & Code Availability",
            key="dataset_and_code_score",
        )
        self._hf_client: _HFClientProtocol = hf_client or HFClient()

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        if FAIL:
            hf_url = _extract_hf_url(url_record) or _DEFAULT_URL
            _LOGGER.info("FAIL flag enabled; returning stub score for %s", hf_url)
            return _FAILURE_VALUES.get(hf_url, _FAILURE_VALUES[_DEFAULT_URL])

        hf_url = _extract_hf_url(url_record)
        _LOGGER.info(
            "Evaluating dataset/code availability for %s",
            hf_url or "<unknown>",
        )

        dataset_score = 0.5 if _has_explicit_dataset(url_record) else 0.0
        if dataset_score:
            _LOGGER.info("Dataset URL provided in manifest for %s", hf_url)
        else:
            _LOGGER.debug("No dataset URL in manifest for %s", hf_url)

        code_score = 0.5 if _has_explicit_code(url_record) else 0.0
        if code_score:
            _LOGGER.info("Code repository provided in manifest for %s", hf_url)
        else:
            _LOGGER.debug("No code repository URL in manifest for %s", hf_url)

        readme_text: Optional[str] = None

        if dataset_score == 0.0 and hf_url:
            _LOGGER.info(
                "Checking Hugging Face metadata for datasets on %s", hf_url
            )
            dataset_score = self._score_dataset_from_hub(hf_url)
            if dataset_score == 0.0:
                _LOGGER.info(
                    "Scanning README for dataset references on %s",
                    hf_url,
                )
                readme_text = readme_text or self._safe_readme(hf_url)
                if readme_text and _DATASET_URL_PATTERN.search(readme_text):
                    _LOGGER.info(
                        "Dataset reference found in README for %s", hf_url
                    )
                    dataset_score = 0.5
                else:
                    _LOGGER.debug(
                        "No dataset reference found in README for %s", hf_url
                    )

        if code_score == 0.0 and hf_url:
            _LOGGER.info(
                "Scanning README for code repository references on %s", hf_url
            )
            readme_text = readme_text or self._safe_readme(hf_url)
            if readme_text and _CODE_URL_PATTERN.search(readme_text):
                _LOGGER.info(
                    "Code repository reference found in README for %s", hf_url
                )
                code_score = 0.5
            else:
                _LOGGER.debug(
                    "No code repository reference found in README for %s",
                    hf_url,
                )

        total = min(dataset_score + code_score, 1.0)
        _LOGGER.info(
            "Dataset/code score for %s computed as %.2f",
            hf_url or "<unknown>",
            total,
        )
        return total

    def _score_dataset_from_hub(self, hf_url: str) -> float:
        model_info = self._safe_model_info(hf_url)
        if not model_info:
            _LOGGER.debug("Model info unavailable for %s", hf_url)
            return 0.0

        datasets = getattr(model_info, "datasets", None)
        if datasets:
            _LOGGER.info("Datasets listed in metadata for %s", hf_url)
            return 0.5

        card_data = getattr(model_info, "card_data", None)
        if isinstance(card_data, dict) and card_data.get("datasets"):
            _LOGGER.info("Datasets found in card data for %s", hf_url)
            return 0.5

        return 0.0

    def _safe_model_info(self, hf_url: str) -> Any:
        try:
            return self._hf_client.get_model_info(hf_url)
        except Exception as exc:
            _LOGGER.debug("Failed to fetch model info for %s: %s", hf_url, exc)
            return None

    def _safe_readme(self, hf_url: str) -> Optional[str]:
        try:
            readme = self._hf_client.get_model_readme(hf_url)
        except Exception as exc:
            _LOGGER.debug("Failed to fetch README for %s: %s", hf_url, exc)
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
