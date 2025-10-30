"""Dataset quality metric focusing on documentation and freshness."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol
from urllib.parse import urlparse

from src.clients.hf_client import HFClient
from src.metrics.base import Metric, MetricOutput
from src.utils.env import enable_readme_fallback, fail_stub_active

_LOGGER = logging.getLogger(__name__)

FAIL = False
_DEFAULT_URL = "https://huggingface.co/google-bert/bert-base-uncased"
_FAILURE_VALUES: Dict[str, float] = {
    "https://huggingface.co/google-bert/bert-base-uncased": 0.10,
    "https://huggingface.co/parvk11/audience_classifier_model": 0.62,
    "https://huggingface.co/openai/whisper-tiny/tree/main": 0.63,
}

_DATASET_URL_PATTERN = re.compile(
    r"https?://huggingface\.co/(?:datasets/)?[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
    re.IGNORECASE,
)

METADATA_WEIGHT = 0.40
SPLITS_WEIGHT = 0.20
LICENSE_WEIGHT = 0.15
ADOPTION_WEIGHT = 0.15
FRESHNESS_WEIGHT = 0.10

MIN_EXAMPLES_FOR_SPLIT = 1000
DOWNLOADS_FULL_SCORE = 10000
LIKES_FULL_SCORE = 100
MODELS_FULL_SCORE = 50
FRESHNESS_FULL_DAYS = 180
FRESHNESS_ZERO_DAYS = 1095

OSI_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "gpl-3.0",
    "lgpl-3.0",
    "mpl-2.0",
    "wtfpl",
    "ecl-2.0",
}


class _HFClientProtocol(Protocol):
    def dataset_exists(self, dataset_id: str) -> bool: ...

    def get_dataset_info(self, dataset_id: str) -> Any: ...

    def get_model_info(self, repo_id: str) -> Any: ...

    def get_model_readme(self, repo_id: str) -> str: ...

    def count_models_trained_on_dataset(self, dataset_id: str) -> int: ...


class DatasetQualityMetric(Metric):
    """Quality assessment for datasets referenced by a model."""

    def __init__(self, hf_client: Optional[_HFClientProtocol] = None) -> None:
        super().__init__(name="Dataset Quality", key="dataset_quality")
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
            "Evaluating dataset quality for %s",
            hf_url or "<unknown>",
        )

        dataset_slug = self._find_dataset_slug(url_record, hf_url)
        if not dataset_slug:
            _LOGGER.info(
                "No dataset identified for %s; quality score is 0",
                hf_url,
            )
            return 0.0

        dataset_info = self._safe_dataset_info(dataset_slug)
        if dataset_info is None:
            _LOGGER.info(
                "Dataset info unavailable for %s; quality score is 0",
                dataset_slug,
            )
            return 0.0

        models_count = self._count_models_using_dataset(dataset_slug)

        metadata_score = self._score_metadata(dataset_info)
        splits_score = self._score_splits(dataset_info)
        license_score = self._score_license(dataset_info)
        adoption_score = self._score_adoption(dataset_info, models_count)
        freshness_score = self._score_freshness(dataset_info)

        total = (
            metadata_score
            + splits_score
            + license_score
            + adoption_score
            + freshness_score
        )
        final_score = min(total, 1.0)
        downloads = getattr(dataset_info, "downloads", "?")
        likes = getattr(dataset_info, "likes", "?")
        last_modified = getattr(dataset_info, "last_modified", "unknown")
        _LOGGER.info(
            "Dataset quality sub-scores for %s: metadata=%.2f splits=%.2f "
            "license=%.2f adoption=%.2f freshness=%.2f (downloads=%s, "
            "likes=%s, models_using=%d, last_modified=%s)",
            dataset_slug,
            metadata_score,
            splits_score,
            license_score,
            adoption_score,
            freshness_score,
            downloads,
            likes,
            models_count,
            last_modified,
        )
        _LOGGER.info(
            "Dataset quality score for %s computed as %.2f",
            dataset_slug,
            final_score,
        )
        return final_score

    def _find_dataset_slug(
        self,
        url_record: Dict[str, str],
        hf_url: Optional[str],
    ) -> Optional[str]:
        slug = self._slug_from_manifest(url_record, hf_url)
        if slug:
            return slug

        slug = self._slug_from_metadata(hf_url)
        if slug:
            return slug

        slug = self._slug_from_readme(hf_url)
        if slug:
            return slug

        return None

    def _slug_from_manifest(
        self,
        url_record: Dict[str, str],
        hf_url: Optional[str],
    ) -> Optional[str]:
        ds_url = url_record.get("ds_url")
        if not ds_url:
            return None

        if self._dataset_reference_is_valid(ds_url):
            slug = _to_dataset_slug(ds_url)
            if slug:
                _LOGGER.info(
                    "Dataset slug from manifest for %s: %s",
                    hf_url,
                    slug,
                )
                return slug

        _LOGGER.info(
            "Dataset reference in manifest for %s is invalid: %s",
            hf_url,
            ds_url,
        )
        return None

    def _slug_from_metadata(self, hf_url: Optional[str]) -> Optional[str]:
        if not hf_url:
            return None

        model_info = self._safe_model_info(hf_url)
        if not model_info:
            return None

        entries = _flatten_dataset_entries(
            getattr(model_info, "datasets", None)
        )
        slug = self._first_valid_dataset(entries)
        if slug:
            _LOGGER.info(
                "Dataset slug inferred from metadata for %s: %s",
                hf_url,
                slug,
            )
            return slug

        card_data = getattr(model_info, "card_data", None)
        if isinstance(card_data, dict):
            entries = _flatten_dataset_entries(card_data.get("datasets"))
            slug = self._first_valid_dataset(entries)
            if slug:
                _LOGGER.info(
                    "Dataset slug inferred from card data for %s: %s",
                    hf_url,
                    slug,
                )
                return slug

        return None

    def _slug_from_readme(self, hf_url: Optional[str]) -> Optional[str]:
        if not hf_url:
            return None
        if not enable_readme_fallback():
            _LOGGER.info(
                "Dataset slug README fallback disabled for %s",
                hf_url,
            )
            return None

        readme = self._safe_readme(hf_url)
        candidate_url = _extract_dataset_from_readme(readme or "")
        if candidate_url and self._dataset_reference_is_valid(candidate_url):
            slug = _to_dataset_slug(candidate_url)
            if slug:
                _LOGGER.info(
                    "Dataset slug inferred from README for %s: %s",
                    hf_url,
                    slug,
                )
                return slug

        return None

    def _first_valid_dataset(self, entries: list[str]) -> Optional[str]:
        for entry in entries:
            if self._dataset_reference_is_valid(entry):
                slug = _to_dataset_slug(entry)
                if slug:
                    return slug
        return None

    def _score_metadata(self, dataset_info: Any) -> float:
        card_data = getattr(dataset_info, "card_data", {}) or {}
        description = getattr(dataset_info, "description", None)

        signals = 0
        total_signals = 6

        if description:
            signals += 1

        for key in (
            "annotations_creators",
            "language",
            "license",
            "task_categories",
            "dataset_info",
        ):
            if card_data.get(key):
                signals += 1

        score = METADATA_WEIGHT * (signals / total_signals)
        _LOGGER.debug(
            "Metadata score derived as %.3f (%d/%d signals)",
            score,
            signals,
            total_signals,
        )
        return score

    def _score_splits(self, dataset_info: Any) -> float:
        card_data = getattr(dataset_info, "card_data", {}) or {}
        dataset_info_section = card_data.get("dataset_info", {})
        splits = None
        if isinstance(dataset_info_section, dict):
            splits = dataset_info_section.get("splits")
        if not splits:
            return 0.0

        score = SPLITS_WEIGHT * 0.5
        if any(
            isinstance(split, dict)
            and split.get("num_examples", 0) >= MIN_EXAMPLES_FOR_SPLIT
            for split in splits
        ):
            score = SPLITS_WEIGHT

        _LOGGER.debug("Split score computed as %.3f", score)
        return score

    def _score_license(self, dataset_info: Any) -> float:
        card_data = getattr(dataset_info, "card_data", {}) or {}
        license_field = card_data.get("license")
        licenses: list[str] = []
        if isinstance(license_field, str):
            licenses = [license_field.lower()]
        elif isinstance(license_field, list):
            licenses = [str(value).lower() for value in license_field]

        if any(lic in OSI_LICENSES for lic in licenses):
            score = LICENSE_WEIGHT
        elif licenses:
            score = LICENSE_WEIGHT * 0.5
        else:
            score = 0.0

        _LOGGER.debug("License score computed as %.3f", score)
        return score

    def _score_adoption(
        self,
        dataset_info: Any,
        models_count: int,
    ) -> float:
        downloads = getattr(dataset_info, "downloads", 0) or 0
        likes = getattr(dataset_info, "likes", 0) or 0

        download_component = min(downloads / DOWNLOADS_FULL_SCORE, 1.0)
        likes_component = min(likes / LIKES_FULL_SCORE, 1.0)
        models_component = min(models_count / MODELS_FULL_SCORE, 1.0)

        score = ADOPTION_WEIGHT * (
            download_component + likes_component + models_component
        ) / 3
        _LOGGER.debug(
            "Adoption score=%.3f (downloads=%d, likes=%d, models=%d)",
            score,
            downloads,
            likes,
            models_count,
        )
        return score

    def _score_freshness(self, dataset_info: Any) -> float:
        last_modified = getattr(dataset_info, "last_modified", None)
        if not isinstance(last_modified, datetime):
            return 0.0

        if last_modified.tzinfo is None:
            last_modified = last_modified.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age_days = (now - last_modified).days
        if age_days <= FRESHNESS_FULL_DAYS:
            normalized = 1.0
        elif age_days >= FRESHNESS_ZERO_DAYS:
            normalized = 0.0
        else:
            normalized = 1 - (
                (age_days - FRESHNESS_FULL_DAYS)
                / (FRESHNESS_ZERO_DAYS - FRESHNESS_FULL_DAYS)
            )

        score = FRESHNESS_WEIGHT * max(0.0, min(normalized, 1.0))
        _LOGGER.debug(
            "Freshness score computed as %.3f (age_days=%d)",
            score,
            age_days,
        )
        return score

    def _dataset_score_from_manifest(
        self,
        url_record: Dict[str, str],
        hf_url: Optional[str],
    ) -> float:
        ds_url = url_record.get("ds_url")
        if not ds_url or not ds_url.strip():
            _LOGGER.debug("No dataset URL in manifest for %s", hf_url)
            return 0.0

        if self._dataset_reference_is_valid(ds_url):
            _LOGGER.info(
                "Dataset URL provided in manifest for %s: %s",
                hf_url,
                ds_url,
            )
            return 0.5

        _LOGGER.info(
            "Dataset URL in manifest for %s did not resolve: %s",
            hf_url,
            ds_url,
        )
        return 0.0

    def _count_models_using_dataset(self, slug: str) -> int:
        try:
            return self._hf_client.count_models_trained_on_dataset(slug)
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.debug(
                "Counting models trained on %s failed: %s",
                slug,
                exc,
            )
            return 0

    def _dataset_reference_is_valid(self, reference: str) -> bool:
        slug = _to_dataset_slug(reference)
        if not slug:
            return False
        try:
            return self._hf_client.dataset_exists(slug)
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.debug("Dataset validation failed for %s: %s", slug, exc)
            return False

    def _safe_dataset_info(self, dataset_slug: str) -> Optional[Any]:
        try:
            return self._hf_client.get_dataset_info(dataset_slug)
        except Exception as exc:
            _LOGGER.debug(
                "Failed to fetch dataset info for %s: %s",
                dataset_slug,
                exc,
            )
            return None

    def _safe_model_info(self, hf_url: str) -> Optional[Any]:
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
