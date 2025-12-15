"""ACMEModels Repository

Introductory remarks: This module implements the performance-claims metric,
which detects explicit benchmark/evaluation evidence in Hugging Face metadata
or README text (with optional LLM fallback).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Protocol, Sequence

from src.clients.hf_client import HFClient
from src.config import (LLM_ANALYSIS_MODEL, LLM_EXTRACTION_MODEL,
                        LLM_TEMPERATURE)
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

ANALYSIS_SYSTEM_PROMPT = (
    "You are an AI assistant that analyses Hugging Face model cards for "
    "evidence of performance claims. Reason aloud before answering."
)
ANALYSIS_USER_TEMPLATE = (
    "Consider the model card below. Determine whether it contains "
    "explicit evaluation results such as benchmark names, datasets, or "
    "metrics with numeric values. Provide a step-by-step justification and "
    "finish with 'Final answer: YES' or 'Final answer: NO'.\n\n"
    "Model card:\n{readme}\n"
)

EXTRACTION_SYSTEM_PROMPT = (
    "You read analyses and output a concise YES/NO decision."
)
EXTRACTION_USER_TEMPLATE = (
    "Given the analysis below, respond with exactly 'YES' or 'NO' to "
    "indicate whether performance claims are present. Return only the word."
    "\n\nAnalysis:\n{analysis}\n"
)


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


class _PurdueClientProtocol(Protocol):
    """
    _PurdueClientProtocol: Class description.
    """

    def llm(
        self,
        prompt: Optional[str] = None,
        *,
        messages: Optional[Sequence[Dict[str, str]]] = None,
        model: str = LLM_ANALYSIS_MODEL,
        stream: bool = False,
        temperature: float = LLM_TEMPERATURE,
        **extra: Any,
    ) -> str: ...


class PerformanceMetric(Metric):
    """Binary performance-claims detector using metadata or an LLM."""

    def __init__(
        self,
        hf_client: Optional[_HFClientProtocol] = None,
        purdue_client: Optional[_PurdueClientProtocol] = None,
    ) -> None:
        """
        __init__: Function description.
        :param hf_client:
        :param purdue_client:
        :returns:
        """

        super().__init__(name="Performance Claims", key="performance_claims")
        self._hf = hf_client or HFClient()
        self._purdue_client: Optional[_PurdueClientProtocol] = purdue_client

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        """Return `1.0` when benchmark/evaluation claims are detected, else `0.0`."""
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

        if _readme_has_numeric_claims(readme_text):
            _LOGGER.info(
                "Performance metric: README heuristic claims detected for %s",
                hf_url,
            )
            return 1.0

        has_claims = self._llm_detect_claims(readme_text)
        _LOGGER.info(
            "Performance metric LLM result for %s: %s",
            hf_url,
            has_claims,
        )
        return 1.0 if has_claims else 0.0

    def _safe_model_info(self, hf_url: str) -> Optional[Any]:
        """
        _safe_model_info: Function description.
        :param hf_url:
        :returns:
        """

        try:
            return self._hf.get_model_info(hf_url)
        except Exception as exc:
            _LOGGER.debug(
                "Performance metric: model info unavailable: %s",
                exc,
            )
            return None

    def _safe_readme(self, hf_url: str) -> str:
        """
        _safe_readme: Function description.
        :param hf_url:
        :returns:
        """

        try:
            return self._hf.get_model_readme(hf_url)
        except Exception as exc:
            _LOGGER.debug(
                "Performance metric: README unavailable: %s",
                exc,
            )
            return ""

    def _get_purdue_client(self) -> Optional[_PurdueClientProtocol]:
        """
        _get_purdue_client: Function description.
        :param:
        :returns:
        """

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
        """
        _llm_detect_claims: Function description.
        :param readme_text:
        :returns:
        """

        client = self._get_purdue_client()
        if client is None:
            return False

        try:
            analysis = client.llm(
                messages=[
                    {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": ANALYSIS_USER_TEMPLATE.format(
                            readme=readme_text
                        ),
                    },
                ],
                model=LLM_ANALYSIS_MODEL,
                temperature=LLM_TEMPERATURE,
            )
            decision = client.llm(
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": EXTRACTION_USER_TEMPLATE.format(
                            analysis=analysis
                        ),
                    },
                ],
                model=LLM_EXTRACTION_MODEL,
                temperature=LLM_TEMPERATURE,
            )
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
    """Return True when model metadata includes structured evaluation results."""
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
    """Parse Hugging Face `model-index` entries into a list of metric claims."""
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
    """Extract the Hugging Face URL from an input record, if present."""
    return record.get("hf_url")


_METRIC_VALUE_PATTERN = re.compile(
    r"(?is)\b("
    r"accuracy|acc|f1(?:-score)?|bleu|rouge(?:-[0-9l])?|wer|cer|"
    r"mrr|ndcg|precision|recall|"
    r"top[- ]?1|top[- ]?5|"
    r"map|mAP|"
    r"perplexity|ppl"
    r")\b"
    r"[^\n0-9]{0,30}"
    r"([0-9]{1,4}(?:\.[0-9]+)?)"
    r"%?"
)

_VALUE_METRIC_PATTERN = re.compile(
    r"(?is)"
    r"([0-9]{1,4}(?:\.[0-9]+)?)"
    r"%?"
    r"[^\nA-Za-z]{0,20}"
    r"\b("
    r"accuracy|acc|f1(?:-score)?|bleu|rouge(?:-[0-9l])?|wer|cer|"
    r"mrr|ndcg|precision|recall|"
    r"top[- ]?1|top[- ]?5|"
    r"map|mAP|"
    r"perplexity|ppl"
    r")\b"
)


def _readme_has_numeric_claims(readme_text: str) -> bool:
    """Heuristic detector for benchmark
    claims when structured metadata is absent.

    Avoids relying on the Purdue LLM client by detecting common metric/value
    patterns (e.g. "F1: 89.5", "92% accuracy", "| BLEU | 27.1 |").
    """

    if not readme_text:
        return False
    if not re.search(r"\d", readme_text):
        return False

    text = re.sub(r"```.*?```", "", readme_text, flags=re.DOTALL)
    candidates: list[tuple[str, str]] = []
    for match in _METRIC_VALUE_PATTERN.finditer(text):
        candidates.append((match.group(1), match.group(2)))
    for match in _VALUE_METRIC_PATTERN.finditer(text):
        candidates.append((match.group(2), match.group(1)))

    for metric_raw, value_raw in candidates:
        metric = metric_raw.strip().lower()
        try:
            value = float(value_raw)
        except ValueError:
            continue

        # Ignore years and other likely non-metric numbers.
        if 1900 <= value <= 2100:
            continue

        if metric in {"perplexity", "ppl"}:
            if value <= 0:
                continue
            return True

        if 0 <= value <= 100:
            return True

    return False
