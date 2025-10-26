from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Protocol, Sequence

from src.clients.hf_client import HFClient
from src.metrics.base import Metric, MetricOutput
from src.utils.env import enable_readme_fallback, fail_stub_active

_LOGGER = logging.getLogger(__name__)

FAIL = False

_DEFAULT_URL = "https://huggingface.co/google-bert/bert-base-uncased"

_FAILURE_VALUES: Dict[str, float] = {
    "https://huggingface.co/google-bert/bert-base-uncased": 0.0,
    "https://huggingface.co/parvk11/audience_classifier_model": 1.0,
    "https://huggingface.co/openai/whisper-tiny/tree/main": 0.0,
}


class _HFClientProtocol(Protocol):
    def get_model_info(self, repo_id: str) -> Any: ...

    def get_model_readme(self, repo_id: str) -> str: ...


class _PurdueClientProtocol(Protocol):
    def llm(
        self,
        prompt: str,
        *,
        model: str = "llama3.1:latest",
        stream: bool = False,
        **extra: Any,
    ) -> str: ...


class PerformanceMetric(Metric):
    """Binary performance-claims detector using metadata or an LLM."""

    def __init__(
        self,
        hf_client: Optional[_HFClientProtocol] = None,
        purdue_client: Optional[_PurdueClientProtocol] = None,
    ) -> None:
        super().__init__(name="Performance Claims", key="performance_claims")
        self._hf = hf_client or HFClient()
        self._purdue_client: Optional[_PurdueClientProtocol] = purdue_client

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        hf_url = _extract_hf_url(url_record)
        if fail_stub_active(FAIL):
            time.sleep(0.05)
            url = hf_url or _DEFAULT_URL
            return _FAILURE_VALUES.get(url, _FAILURE_VALUES[_DEFAULT_URL])

        if not hf_url:
            return 0.0

        info = self._safe_model_info(hf_url)
        if _has_structured_claims(info):
            _LOGGER.info(
                "Performance metric: structured claims detected for %s",
                hf_url,
            )
            return 1.0

        if not enable_readme_fallback():
            _LOGGER.info(
                "Performance metric: README fallback disabled for %s",
                hf_url,
            )
            return 0.0

        readme_text = self._safe_readme(hf_url)
        if not readme_text.strip():
            _LOGGER.info(
                "Performance metric: empty README for %s",
                hf_url,
            )
            return 0.0

        has_claims = self._llm_detect_claims(readme_text)
        _LOGGER.info(
            "Performance metric LLM result for %s: %s",
            hf_url,
            has_claims,
        )
        return 1.0 if has_claims else 0.0

    def _safe_model_info(self, hf_url: str) -> Optional[Any]:
        try:
            return self._hf.get_model_info(hf_url)
        except Exception as exc:
            _LOGGER.debug(
                "Performance metric: model info unavailable: %s",
                exc,
            )
            return None

    def _safe_readme(self, hf_url: str) -> str:
        try:
            return self._hf.get_model_readme(hf_url)
        except Exception as exc:
            _LOGGER.debug(
                "Performance metric: README unavailable: %s",
                exc,
            )
            return ""

    def _get_purdue_client(self) -> Optional[_PurdueClientProtocol]:
        if self._purdue_client is not None:
            return self._purdue_client

        try:
            from src.clients.purdue_client import PurdueClient

            self._purdue_client = PurdueClient()
        except Exception as exc:
            _LOGGER.info(
                "Performance metric: Purdue client unavailable (%s)",
                exc,
            )
            self._purdue_client = None
        return self._purdue_client

    def _llm_detect_claims(self, readme_text: str) -> bool:
        client = self._get_purdue_client()
        if client is None:
            return False

        analysis_prompt = (
            "You are reviewing a Hugging Face model card. Determine whether "
            "it contains explicit performance claims such as benchmark "
            "names, datasets, or metrics with numeric results. Think "
            "step-by-step and end with 'Final answer: YES' or 'Final answer: "
            "NO'.\n\nModel card:\n"
            f"{readme_text}\n"
        )
        extraction_prompt = (
            "Based on the analysis, respond with only YES or NO to indicate "
            "whether performance claims are present.\n\n"
            "Analysis:\n{analysis}\n"
        )

        try:
            analysis = client.llm(analysis_prompt)
            decision = client.llm(extraction_prompt.format(analysis=analysis))
            match = re.search(r"\b(YES|NO)\b", decision.upper())
            if match:
                return match.group(1) == "YES"
            _LOGGER.debug(
                "Performance metric: LLM extraction yielded no YES/NO: %s",
                decision,
            )
        except Exception as exc:
            _LOGGER.debug(
                "Performance metric: LLM detection failed: %s",
                exc,
            )
        return False


def _has_structured_claims(info: Optional[Any]) -> bool:
    if info is None:
        return False

    card = getattr(info, "card_data", None)
    if card is None:
        card = getattr(info, "cardData", None)
    if not isinstance(card, dict):
        return False

    model_index = card.get("model-index")
    if isinstance(model_index, list) and _parse_model_index(model_index):
        return True

    evals = card.get("eval_results") or card.get("evaluation")
    if isinstance(evals, list):
        for entry in evals:
            if isinstance(entry, dict):
                metric = entry.get("metric")
                value = entry.get("value")
                if metric and value is not None:
                    return True

    metrics_field = card.get("metrics")
    if isinstance(metrics_field, list) and metrics_field:
        return True

    benchmark = card.get("benchmark")
    if benchmark:
        return True

    return False


def _parse_model_index(model_index: Sequence[Any]) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    for entry in model_index:
        if not isinstance(entry, dict):
            continue
        results = entry.get("results")
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            metrics = result.get("metrics")
            if not isinstance(metrics, list):
                continue
            for metric_entry in metrics:
                if not isinstance(metric_entry, dict):
                    continue
                metric = metric_entry.get("type") or metric_entry.get("name")
                value = metric_entry.get("value")
                dataset = result.get("dataset") or {}
                dataset_name = None
                if isinstance(dataset, dict):
                    dataset_name = dataset.get("name") or dataset.get("type")
                if metric and value is not None:
                    claims.append(
                        {
                            "metric": metric,
                            "value": value,
                            "dataset": dataset_name,
                        }
                    )
    return claims


def _extract_hf_url(record: Dict[str, str]) -> Optional[str]:
    return record.get("hf_url")
