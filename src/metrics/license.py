"""License compatibility metric for Hugging Face models."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Protocol

from src.clients.hf_client import HFClient
from src.metrics.base import Metric, MetricOutput
from src.services.license_analysis import (collect_hf_license_candidates,
                                           evaluate_classification,
                                           load_license_policy,
                                           normalize_license_candidates)
from src.utils.env import fail_stub_active

_LOGGER = logging.getLogger(__name__)

FAIL = False

_DEFAULT_URL = "https://huggingface.co/google-bert/bert-base-uncased"

_FAILURE_VALUES: Dict[str, float] = {
    "https://huggingface.co/google-bert/bert-base-uncased": 0.1,
    "https://huggingface.co/parvk11/audience_classifier_model": 0.8,
    "https://huggingface.co/openai/whisper-tiny/tree/main": 0.9,
}

COMPAT_WEIGHT = 0.8
CLARITY_WEIGHT = 0.2

class _HFClientProtocol(Protocol):
    def get_model_info(self, repo_id: str) -> Any: ...

    def get_model_readme(self, repo_id: str) -> str: ...


class LicenseMetric(Metric):
    """License clarity and compatibility score.

    Uses policy files under ``data/`` to categorize SPDX-like license
    identifiers. Prefers HF metadata; falls back to README parsing.
    """

    def __init__(self, hf_client: Optional[_HFClientProtocol] = None) -> None:
        super().__init__(name="License Score", key="license")
        self._hf: _HFClientProtocol = hf_client or HFClient()

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        hf_url = _extract_hf_url(url_record)
        if fail_stub_active(FAIL):
            time.sleep(0.05)
            url = hf_url or _DEFAULT_URL
            return _FAILURE_VALUES.get(url, _FAILURE_VALUES[_DEFAULT_URL])

        if not hf_url:
            return 0.0

        policy = load_license_policy()
        from_meta, from_readme = collect_hf_license_candidates(
            self._hf, hf_url
        )
        candidates = normalize_license_candidates([*from_meta, *from_readme])

        _LOGGER.info(
            "License metric inputs for %s: metadata=%s readme=%s "
            "normalized=%s",
            hf_url,
            from_meta,
            from_readme,
            candidates,
        )

        if not candidates:
            _LOGGER.info(
                "License metric: no recognized licenses for %s",
                hf_url,
            )
            return 0.0

        classification = evaluate_classification(candidates, policy)
        compat = 0.0
        if classification == "compatible":
            compat = 1.0
        elif classification == "caution":
            compat = 0.5
        elif classification == "incompatible":
            compat = 0.0
        else:
            compat = 0.0

        final = max(0.0, min(compat, 1.0))
        _LOGGER.info(
            "License metric for %s: class=%s score=%.2f",
            hf_url,
            classification,
            final,
        )
        return final


def _extract_hf_url(record: Dict[str, str]) -> Optional[str]:
    return record.get("hf_url")


