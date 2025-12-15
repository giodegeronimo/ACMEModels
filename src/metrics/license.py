"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

License compatibility metric for Hugging Face models.
"""

from __future__ import annotations

import logging
import re
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


class _HFClientProtocol(Protocol):
    """
    get_model_info: Function description.
    :param repo_id:
    :returns:
    """

    """
    _HFClientProtocol: Class description.
    """

    def get_model_info(self, repo_id: str) -> Any: ...

    """
    get_model_readme: Function description.
    :param repo_id:
    :returns:
    """

    def get_model_readme(self, repo_id: str) -> str: ...


class LicenseMetric(Metric):
    """License clarity and compatibility score.

    Uses policy files under ``data/`` to categorize SPDX-like license
    identifiers. Prefers HF metadata; falls back to README parsing.
    """

    def __init__(self, hf_client: Optional[_HFClientProtocol] = None) -> None:
        """
        __init__: Function description.
        :param hf_client:
        :returns:
        """

        super().__init__(name="License Score", key="license")
        self._hf: _HFClientProtocol = hf_client or HFClient()

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        """
        compute: Function description.
        :param url_record:
        :returns:
        """

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
    """
    _extract_hf_url: Function description.
    :param record:
    :returns:
    """

    return record.get("hf_url")


def _normalize_slug(value: str) -> Optional[str]:
    """
    _normalize_slug: Function description.
    :param value:
    :returns:
    """

    s = value.strip().lower()
    if not s:
        return None
    synonyms: Dict[str, str] = {
        "mit license": "MIT",
        "mit": "MIT",
        "apache license 2.0": "Apache-2.0",
        "apache license, version 2.0": "Apache-2.0",
        "apache2": "Apache-2.0",
        "apache 2.0": "Apache-2.0",
        "apache-2.0": "Apache-2.0",
        "bsd 3-clause": "BSD-3-Clause",
        'bsd 3-clause "new" or "revised" license': "BSD-3-Clause",
        "bsd-3-clause": "BSD-3-Clause",
        "bsd 2-clause": "BSD-2-Clause",
        'bsd 2-clause "simplified" license': "BSD-2-Clause",
        "bsd-2-clause": "BSD-2-Clause",
        "mozilla public license 2.0": "MPL-2.0",
        "mpl 2.0": "MPL-2.0",
        "mpl-2.0": "MPL-2.0",
        "gnu lesser general public license v2.1": "LGPL-2.1-only",
        "lgpl 2.1": "LGPL-2.1-only",
        "lgpl v2.1": "LGPL-2.1-only",
        "lgpl-2.1": "LGPL-2.1-only",
        "gnu lesser general public license v3.0": "LGPL-3.0-only",
        "lgpl3": "LGPL-3.0-only",
        "lgpl v3": "LGPL-3.0-only",
        "lgpl-3.0": "LGPL-3.0-only",
        "gnu general public license v3.0": "GPL-3.0-only",
        "gpl v3": "GPL-3.0-only",
        "gpl-3.0": "GPL-3.0-only",
        "gnu affero general public license v3.0": "AGPL-3.0-only",
        "agpl-3.0": "AGPL-3.0-only",
        "cc-by 4.0": "CC-BY-4.0",
        "creative commons attribution 4.0 international": "CC-BY-4.0",
        "cc-by-4.0": "CC-BY-4.0",
        "creative commons zero v1.0 universal": "CC0-1.0",
        "cc0": "CC0-1.0",
        "cc0-1.0": "CC0-1.0",
        "cc-by-sa 4.0": "CC-BY-SA-4.0",
        "creative commons attribution share alike 4.0 international":
            "CC-BY-SA-4.0",
        "cc-by-sa-4.0": "CC-BY-SA-4.0",
        "creative commons attribution-noncommercial 4.0 international":
            "CC-BY-NC-4.0",
        "cc-by-nc-4.0": "CC-BY-NC-4.0",
        "creative commons attribution-noderivatives 4.0 international":
            "CC-BY-ND-4.0",
        "cc-by-nd-4.0": "CC-BY-ND-4.0",
        "openrail-m": "OpenRAIL-M",
        "openrail++": "OpenRAIL++",
        "custom": "Custom",
    }
    if s in synonyms:
        return synonyms[s]
    spdx_like = re.sub(r"\s+", "-", value.strip())
    return spdx_like
