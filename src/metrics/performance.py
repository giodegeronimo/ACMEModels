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
    "https://huggingface.co/google-bert/bert-base-uncased": 0.31,
    "https://huggingface.co/parvk11/audience_classifier_model": 0.99,
    "https://huggingface.co/openai/whisper-tiny/tree/main": 0.11,
}

_CLAIM_PATTERNS = (
    r"accuracy",
    r"f1",
    r"bleu",
    r"wer",
    r"mrr",
    r"rouge",
    r"exact match",
    r"top-1",
    r"top-5",
    r"cer",
    r"precision",
    r"recall",
    r"auc",
    r"mae",
)


class _HFClientProtocol(Protocol):
    def get_model_info(self, repo_id: str) -> Any: ...

    def get_model_readme(self, repo_id: str) -> str: ...


class PerformanceMetric(Metric):
    """Score strength of performance claims for a model."""

    def __init__(
        self,
        hf_client: Optional[_HFClientProtocol] = None,
    ) -> None:
        super().__init__(name="Performance Claims", key="performance_claims")
        self._hf = hf_client or HFClient()

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        hf_url = _extract_hf_url(url_record)
        if fail_stub_active(FAIL):
            time.sleep(0.05)
            url = hf_url or _DEFAULT_URL
            return _FAILURE_VALUES.get(url, _FAILURE_VALUES[_DEFAULT_URL])

        if not hf_url:
            return 0.0

        info = self._safe_model_info(hf_url)
        readme_text = self._safe_readme(hf_url)

        metadata_score, metadata_details = _metadata_claims_score(info)
        readme_score, readme_details = _readme_claims_score(readme_text)
        reproducibility_score, repro_details = _repro_score(info, readme_text)

        components: List[tuple[float, float, str, Dict[str, Any]]] = []
        if metadata_score > 0:
            components.append(
                (0.6, metadata_score, "metadata", metadata_details)
            )
        if readme_score > 0:
            components.append(
                (0.3, readme_score, "readme", readme_details)
            )
        if reproducibility_score > 0:
            components.append(
                (0.1, reproducibility_score, "repro", repro_details)
            )

        if not components:
            if enable_readme_fallback():
                _LOGGER.info(
                    "Performance metric: no structured claims for %s",
                    hf_url,
                )
            else:
                _LOGGER.info(
                    "Performance metric: README fallback disabled and no "
                    "claims for %s",
                    hf_url,
                )
            return 0.0

        weight_sum = sum(weight for weight, _, _, _ in components)
        final_score = 0.0
        for weight, score, label, details in components:
            normalized_weight = weight / weight_sum if weight_sum else 0.0
            final_score += normalized_weight * score
            _LOGGER.info(
                "Performance metric component %s: weight=%.2f score=%.2f "
                "details=%s",
                label,
                normalized_weight,
                score,
                details,
            )

        final_score = max(0.0, min(1.0, final_score))
        _LOGGER.info(
            "Performance metric for %s computed as %.2f",
            hf_url,
            final_score,
        )
        return final_score

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


def _metadata_claims_score(
    info: Optional[Any],
) -> tuple[float, Dict[str, Any]]:
    if info is None:
        return 0.0, {}

    card = getattr(info, "card_data", None)
    if card is None:
        card = getattr(info, "cardData", None)
    if not isinstance(card, dict):
        return 0.0, {}

    model_index = card.get("model-index")
    if isinstance(model_index, list):
        claims = _parse_model_index(model_index)
        if claims:
            _LOGGER.info(
                "Performance metadata claims: %s",
                claims,
            )
            return 1.0, {"model_index": claims}

    evals = card.get("eval_results") or card.get("evaluation")
    if isinstance(evals, list):
        claims = []
        for entry in evals:
            if not isinstance(entry, dict):
                continue
            metric = entry.get("metric")
            value = entry.get("value")
            dataset = entry.get("dataset")
            if metric and value is not None:
                claims.append(
                    {
                        "metric": metric,
                        "value": value,
                        "dataset": dataset,
                    }
                )
        if claims:
            return 0.7, {"eval_results": claims}

    metrics_field = card.get("metrics")
    if isinstance(metrics_field, list) and metrics_field:
        return 0.5, {"metrics": metrics_field}

    benchmark = card.get("benchmark")
    if benchmark:
        return 0.3, {"benchmark": benchmark}

    return 0.0, {}


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


def _readme_claims_score(readme_text: str) -> tuple[float, Dict[str, Any]]:
    if not readme_text or not enable_readme_fallback():
        return 0.0, {}

    tables = _extract_tables(readme_text)
    numeric_claims = _extract_numeric_claims(readme_text)

    if tables:
        _LOGGER.info("Performance README tables found: %d", len(tables))
        return 0.9, {"tables": tables[:3]}

    if numeric_claims:
        return 0.6, {"claims": numeric_claims[:5]}

    if re.search(r"benchmark|evaluation|results", readme_text, re.IGNORECASE):
        return 0.3, {"mentions": "benchmark keywords"}

    return 0.0, {}


def _repro_score(
    info: Optional[Any], readme_text: str
) -> tuple[float, Dict[str, Any]]:
    signals: Dict[str, Any] = {}
    score = 0.0

    if info is not None:
        card = getattr(info, "card_data", None) or {}
        if isinstance(card, dict):
            paper = card.get("paper") or card.get("arxiv")
            if paper:
                signals["paper"] = paper
                score += 0.4
            scripts = card.get("training") or card.get("eval")
            if scripts:
                signals["scripts"] = scripts
                score += 0.3

    if readme_text:
        if re.search(r"eval\.(py|sh)|evaluation", readme_text, re.IGNORECASE):
            signals.setdefault("readme_scripts", True)
            score += 0.3
        if re.search(r"arxiv|paper|doi", readme_text, re.IGNORECASE):
            signals.setdefault("readme_paper", True)
            score += 0.2

    score = min(1.0, score)
    return score, signals


def _extract_tables(readme_text: str) -> List[str]:
    tables: List[str] = []
    pattern = re.compile(
        r"^\|.+\|\s*$\n(?:^\|(?:[-:]+\|)+\s*$\n)?(?:^\|.+\|\s*$\n)+",
        re.MULTILINE,
    )
    for match in pattern.finditer(readme_text):
        tables.append(match.group(0)[:300])
    return tables


def _extract_numeric_claims(readme_text: str) -> List[str]:
    claims: List[str] = []
    lines = readme_text.splitlines()
    claim_pattern = re.compile(
        rf"(\b(?:{'|'.join(_CLAIM_PATTERNS)})\b.*?\d+\.?\d*)",
        re.IGNORECASE,
    )
    for line in lines:
        match = claim_pattern.search(line)
        if match:
            claims.append(match.group(1).strip())
    return claims


def _extract_hf_url(record: Dict[str, str]) -> Optional[str]:
    return record.get("hf_url")
