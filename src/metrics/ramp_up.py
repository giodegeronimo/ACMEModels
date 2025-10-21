from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, Optional, Protocol

from src.clients.hf_client import HFClient
from src.clients.purdue_client import PurdueClient
from src.metrics.base import Metric, MetricOutput
from src.utils.env import fail_stub_active

_LOGGER = logging.getLogger(__name__)

FAIL = False
_DEFAULT_URL = "https://huggingface.co/google-bert/bert-base-uncased"
_FAILURE_VALUES: Dict[str, float] = {
    "https://huggingface.co/google-bert/bert-base-uncased": 0.11,
    "https://huggingface.co/parvk11/audience_classifier_model": 0.8,
    "https://huggingface.co/openai/whisper-tiny/tree/main": 0.1,
}

_HEADING_PATTERN = re.compile(r"^#{2,}\s+\S+", re.MULTILINE)
_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python|bash|shell)?\n.*?```",
    re.DOTALL,
)
_USAGE_PATTERN = re.compile(
    r"(pipeline\(|AutoModel|AutoTokenizer|from_pretrained)",
    re.IGNORECASE,
)
_EXTERNAL_LINK_PATTERN = re.compile(r"https?://[^\s)]+")

README_LENGTH_THRESHOLD = 400
CODE_BLOCK_THRESHOLD = 2
EXTERNAL_LINK_THRESHOLD = 2
LLM_MIN_LENGTH = 500

METADATA_WEIGHT = 0.40
USAGE_WEIGHT = 0.30
ARTIFACT_WEIGHT = 0.20
EXTERNAL_LINK_WEIGHT = 0.10


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


class RampUpMetric(Metric):
    """Estimate how quickly a developer can start using the model."""

    def __init__(
        self,
        hf_client: Optional[_HFClientProtocol] = None,
        purdue_client: Optional[_PurdueClientProtocol] = None,
    ) -> None:
        super().__init__(name="Ramp-up Score", key="ramp_up_time")
        self._hf_client: _HFClientProtocol = hf_client or HFClient()
        self._purdue_client: Optional[_PurdueClientProtocol] = purdue_client

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
            "Evaluating ramp-up score for %s",
            hf_url or "<unknown>",
        )

        model_info = self._safe_model_info(hf_url) if hf_url else None
        readme_text = self._safe_readme(hf_url) if hf_url else ""
        normalised_readme = readme_text or ""

        metadata_score = self._score_readme_depth(normalised_readme)
        usage_score = self._score_usage_examples(normalised_readme)
        artifact_score = self._score_artifacts(model_info)
        link_score = self._score_external_links(normalised_readme)

        total = metadata_score + usage_score + artifact_score + link_score
        final_score = min(total, 1.0)
        _LOGGER.info(
            "Ramp-up score for %s computed as %.2f",
            hf_url or "<unknown>",
            final_score,
        )
        return final_score

    def _score_readme_depth(self, readme_text: str) -> float:
        if not readme_text:
            _LOGGER.info("README absent or empty; metadata score = 0.0")
            return 0.0

        llm_result = self._score_readme_with_llm(readme_text)
        if llm_result is not None:
            score = METADATA_WEIGHT * llm_result
            _LOGGER.info("LLM metadata score %.3f", score)
            return score

        fallback = self._fallback_readme_score(readme_text)
        _LOGGER.info("Fallback metadata score %.3f", fallback)
        return fallback

    def _score_readme_with_llm(self, readme_text: str) -> Optional[float]:
        client = self._get_purdue_client()
        if client is None:
            return None
        if len(readme_text) < LLM_MIN_LENGTH:
            _LOGGER.debug(
                "Skipping LLM eval; README length %d < %d",
                len(readme_text),
                LLM_MIN_LENGTH,
            )
            return None

        analysis_prompt = (
            "You are evaluating the documentation quality of a machine "
            "learning model README. Think step-by-step about clarity, setup "
            "instructions, usage examples, troubleshooting, and completeness. "
            "Rate quality between 0 and 1 inclusive. Provide detailed "
            "reasoning and end with a line 'Final rating: <number>'.\n\n"
            "README:\n"
            f"{readme_text}\n"
        )
        extraction_prompt = (
            "Extract the numeric rating (between 0 and 1) from the analysis. "
            "Reply with only the number.\n\nAnalysis:\n{analysis}\n"
        )

        try:
            analysis = client.llm(analysis_prompt)
            extraction = client.llm(
                extraction_prompt.format(analysis=analysis)
            )
            match = re.search(r"([01](?:\.\d+)?)", extraction)
            if match:
                value = float(match.group(1))
                return max(0.0, min(value, 1.0))
            _LOGGER.debug("LLM extraction produced no number: %s", extraction)
        except Exception as exc:
            _LOGGER.debug("LLM evaluation failed: %s", exc)
        return None

    def _fallback_readme_score(self, readme_text: str) -> float:
        tokens = _tokenize(readme_text)
        if not tokens:
            return 0.0

        length_component = min(
            len(readme_text) / README_LENGTH_THRESHOLD,
            1.0,
        )
        heading_component = min(
            len(_HEADING_PATTERN.findall(readme_text)) / 6,
            1.0,
        )

        unique_ratio = len(set(tokens)) / len(tokens)
        entropy = _shannon_entropy(tokens)
        max_entropy = (
            math.log2(len(set(tokens))) if len(set(tokens)) > 1 else 0.0
        )
        entropy_component = (
            entropy / max_entropy if max_entropy > 0 else 0.0
        )

        combined = (
            0.4 * length_component
            + 0.3 * heading_component
            + 0.3 * min((unique_ratio + entropy_component) / 2, 1.0)
        )
        return METADATA_WEIGHT * min(combined, 1.0)

    def _score_usage_examples(self, readme_text: str) -> float:
        if not readme_text:
            _LOGGER.info("No README text; usage example score = 0.0")
            return 0.0

        code_blocks = len(_CODE_BLOCK_PATTERN.findall(readme_text))
        usage_hits = len(_USAGE_PATTERN.findall(readme_text))

        code_component = min(code_blocks / CODE_BLOCK_THRESHOLD, 1.0)
        usage_component = 1.0 if usage_hits > 0 else 0.0

        score = USAGE_WEIGHT * (0.6 * code_component + 0.4 * usage_component)
        _LOGGER.info(
            "Usage example score %.3f (blocks=%d, hits=%d)",
            score,
            code_blocks,
            usage_hits,
        )
        return score

    def _get_purdue_client(self) -> Optional[_PurdueClientProtocol]:
        if self._purdue_client is not None:
            return self._purdue_client

        try:
            client = PurdueClient()
        except RuntimeError as exc:
            _LOGGER.info(
                "Purdue client unavailable; skipping LLM evaluation: %s",
                exc,
            )
            self._purdue_client = None
            return None

        self._purdue_client = client
        return client

    def _score_artifacts(self, model_info: Any) -> float:
        if model_info is None:
            _LOGGER.info("Model info unavailable; artifact score = 0.0")
            return 0.0

        siblings = getattr(model_info, "siblings", []) or []
        helpful_files = 0
        helpful_keywords = ("example", "demo", "usage", "notebook", "tutorial")
        for sibling in siblings:
            name = getattr(sibling, "rfilename", "").lower()
            if any(keyword in name for keyword in helpful_keywords):
                helpful_files += 1

        metadata_signals = 0
        if getattr(model_info, "pipeline_tag", None):
            metadata_signals += 1
        if getattr(model_info, "library_name", None):
            metadata_signals += 1

        file_component = min(helpful_files / 2, 1.0)
        metadata_component = metadata_signals / 2

        score = ARTIFACT_WEIGHT * (
            0.7 * file_component + 0.3 * metadata_component
        )
        _LOGGER.info(
            "Artifact score %.3f (files=%d, metadata=%d)",
            score,
            helpful_files,
            metadata_signals,
        )
        return score

    def _score_external_links(self, readme_text: str) -> float:
        if not readme_text:
            _LOGGER.info("No README text; external link score = 0.0")
            return 0.0

        links = _EXTERNAL_LINK_PATTERN.findall(readme_text)
        informative_links: list[str] = []
        allowed_keywords = (
            "docs",
            "huggingface.co/spaces",
            "medium",
            "blog",
        )
        for link in links:
            if any(keyword in link.lower() for keyword in allowed_keywords):
                informative_links.append(link)

        link_component = min(
            len(informative_links) / EXTERNAL_LINK_THRESHOLD,
            1.0,
        )
        score = EXTERNAL_LINK_WEIGHT * link_component
        _LOGGER.info(
            "External link score: %.3f (informative_links=%d)",
            score,
            len(informative_links),
        )
        return score

    def _safe_model_info(self, hf_url: Optional[str]) -> Optional[Any]:
        if not hf_url:
            return None
        try:
            return self._hf_client.get_model_info(hf_url)
        except Exception as exc:
            _LOGGER.debug("Failed to fetch model info for %s: %s", hf_url, exc)
            return None

    def _safe_readme(self, hf_url: Optional[str]) -> Optional[str]:
        if not hf_url:
            return ""
        try:
            readme = self._hf_client.get_model_readme(hf_url)
        except Exception as exc:
            _LOGGER.debug("Failed to fetch README for %s: %s", hf_url, exc)
            return ""
        return readme or ""


def _extract_hf_url(record: Dict[str, str]) -> Optional[str]:
    return record.get("hf_url")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", text.lower())


def _shannon_entropy(tokens: list[str]) -> float:
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy
