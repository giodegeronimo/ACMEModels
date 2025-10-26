from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Protocol
from urllib.parse import urlparse

from src.clients.hf_client import HFClient
from src.metrics.base import Metric, MetricOutput
from src.utils.env import enable_readme_fallback, fail_stub_active

_LOGGER = logging.getLogger(__name__)

FAIL = False
_DEFAULT_URL = "https://huggingface.co/google-bert/bert-base-uncased"
_FAILURE_VALUES: Dict[str, float] = {
    "https://huggingface.co/google-bert/bert-base-uncased": 0.0,
    "https://huggingface.co/parvk11/audience_classifier_model": 0.0,
    "https://huggingface.co/openai/whisper-tiny/tree/main": 1.0,
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

    def dataset_exists(self, dataset_id: str) -> bool: ...


class DatasetAndCodeMetric(Metric):
    """Score availability of datasets and example code for a model."""

    def __init__(self, hf_client: Optional[_HFClientProtocol] = None) -> None:
        super().__init__(
            name="Dataset & Code Availability",
            key="dataset_and_code_score",
        )
        self._hf_client: _HFClientProtocol = hf_client or HFClient()

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        hf_url = _extract_hf_url(url_record)
        if fail_stub_active(FAIL):
            fallback_url = hf_url or _DEFAULT_URL
            _LOGGER.info(
                "FAIL flag enabled; returning stub score for %s",
                fallback_url,
            )
            return _FAILURE_VALUES.get(
                fallback_url,
                _FAILURE_VALUES[_DEFAULT_URL],
            )

        _LOGGER.info(
            "Evaluating dataset/code availability for %s",
            hf_url or "<unknown>",
        )

        dataset_score, dataset_detail = self._dataset_score_from_manifest(
            url_record,
            hf_url,
        )
        dataset_sources: list[str] = []
        if dataset_detail:
            dataset_sources.append(dataset_detail)

        code_score = 0.5 if _has_explicit_code(url_record) else 0.0
        if code_score:
            _LOGGER.info(
                "Code repository provided in manifest for %s: %s",
                hf_url,
                url_record.get("git_url"),
            )
        else:
            _LOGGER.debug("No code repository URL in manifest for %s", hf_url)
        code_sources: list[str] = ["manifest"] if code_score else []

        readme_text: Optional[str] = None

        if dataset_score == 0.0 and hf_url:
            if enable_readme_fallback():
                _LOGGER.info(
                    "Checking Hugging Face metadata for datasets on %s",
                    hf_url,
                )
                dataset_score = self._score_dataset_from_hub(hf_url)
                if dataset_score == 0.0:
                    _LOGGER.info(
                        "Scanning README for dataset references on %s",
                        hf_url,
                    )
                    readme_text = readme_text or self._safe_readme(hf_url)
                    dataset_reference = _extract_dataset_from_readme(
                        readme_text or ""
                    )
                    if dataset_reference and self._dataset_reference_is_valid(
                        dataset_reference
                    ):
                        slug = _to_dataset_slug(dataset_reference)
                        if slug:
                            _LOGGER.info(
                                "Dataset reference found in README for %s: %s",
                                hf_url,
                                f"https://huggingface.co/datasets/{slug}",
                            )
                            dataset_sources.append(f"readme:{slug}")
                        dataset_score = 0.5
                    else:
                        _LOGGER.debug(
                            "No dataset reference found in README for %s",
                            hf_url,
                        )
            else:
                _LOGGER.info(
                    "README-based dataset fallback disabled for %s",
                    hf_url,
                )

        if code_score == 0.0 and hf_url:
            if enable_readme_fallback():
                _LOGGER.info(
                    "Scanning README for code repository references on %s",
                    hf_url,
                )
                readme_text = readme_text or self._safe_readme(hf_url)
                match = (
                    _CODE_URL_PATTERN.search(readme_text)
                    if readme_text
                    else None
                )
                if match:
                    repo_url = match.group(0)
                    _LOGGER.info(
                        "Code repository reference found in README for %s: %s",
                        hf_url,
                        repo_url,
                    )
                    code_score = 0.5
                    code_sources.append("readme")
                else:
                    _LOGGER.debug(
                        "No code repository reference found in README for %s",
                        hf_url,
                    )
            else:
                _LOGGER.info(
                    "README-based code fallback disabled for %s",
                    hf_url,
                )

        total = min(dataset_score + code_score, 1.0)
        _LOGGER.info(
            "Dataset/code score for %s computed as %.2f (dataset=%.2f "
            "sources=%s, code=%.2f sources=%s)",
            hf_url or "<unknown>",
            total,
            dataset_score,
            dataset_sources or ["none"],
            code_score,
            code_sources or ["none"],
        )
        return total

    def _dataset_score_from_manifest(
        self, url_record: Dict[str, str], hf_url: Optional[str]
    ) -> tuple[float, Optional[str]]:
        ds_url = url_record.get("ds_url")
        if not ds_url or not ds_url.strip():
            _LOGGER.debug("No dataset URL in manifest for %s", hf_url)
            return 0.0, None

        if self._dataset_reference_is_valid(ds_url):
            _LOGGER.info(
                "Dataset URL provided in manifest for %s: %s",
                hf_url,
                ds_url,
            )
            slug = _to_dataset_slug(ds_url)
            detail = f"manifest:{slug or ds_url}"
            return 0.5, detail

        _LOGGER.info(
            "Dataset URL in manifest for %s did not resolve: %s",
            hf_url,
            ds_url,
        )
        return 0.0, None

    def _score_dataset_from_hub(self, hf_url: str) -> float:
        model_info = self._safe_model_info(hf_url)
        if not model_info:
            _LOGGER.debug("Model info unavailable for %s", hf_url)
            return 0.0

        dataset_urls = self._collect_dataset_urls(model_info)
        if dataset_urls:
            _LOGGER.info(
                "Datasets listed in metadata for %s: %s",
                hf_url,
                dataset_urls,
            )
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

    def _collect_dataset_urls(self, model_info: Any) -> list[str]:
        urls: list[str] = []

        datasets_attr = getattr(model_info, "datasets", None)
        initial_entries = _flatten_dataset_entries(datasets_attr)
        urls.extend(self._validate_dataset_entries(initial_entries))

        card_data = getattr(model_info, "card_data", None)
        if isinstance(card_data, dict):
            urls.extend(
                self._validate_dataset_entries(
                    _flatten_dataset_entries(card_data.get("datasets"))
                )
            )

        return urls

    def _validate_dataset_entries(self, entries: list[str]) -> list[str]:
        valid_urls: list[str] = []
        for entry in entries:
            if not self._dataset_reference_is_valid(entry):
                continue
            slug = _to_dataset_slug(entry)
            if slug:
                valid_urls.append(f"https://huggingface.co/datasets/{slug}")
        return valid_urls

    def _dataset_reference_is_valid(self, reference: str) -> bool:
        slug = _to_dataset_slug(reference)
        if not slug:
            return False
        try:
            return self._hf_client.dataset_exists(slug)
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.debug("Dataset validation failed for %s: %s", slug, exc)
            return False


def _extract_hf_url(record: Dict[str, str]) -> Optional[str]:
    return record.get("hf_url")


def _has_explicit_dataset(record: Dict[str, str]) -> bool:
    ds_url = record.get("ds_url")
    return bool(ds_url and ds_url.strip())


def _has_explicit_code(record: Dict[str, str]) -> bool:
    git_url = record.get("git_url")
    return bool(git_url and git_url.strip())


def _flatten_dataset_entries(entry: Any) -> list[str]:
    if entry is None:
        return []

    if isinstance(entry, str):
        items = [entry]
    elif isinstance(entry, list):
        items = [item for item in entry if isinstance(item, str)]
    else:
        return []

    return [item.strip() for item in items if item.strip()]


def _extract_dataset_from_readme(text: str) -> Optional[str]:
    if not text:
        return None
    match = _DATASET_URL_PATTERN.search(text)
    if match:
        return match.group(0)
    return None


def _to_dataset_slug(candidate: str) -> Optional[str]:
    if not candidate:
        return None

    if "://" in candidate:
        parsed = urlparse(candidate)
        if parsed.netloc != "huggingface.co":
            return None
        segments = [segment for segment in parsed.path.split("/") if segment]
        if not segments:
            return None
        if segments[0] == "datasets":
            segments = segments[1:]
        if len(segments) < 2:
            return None
        return "/".join(segments[:2])

    if "/" in candidate:
        return candidate

    return None
