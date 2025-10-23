from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Pattern, Protocol

from src.clients.hf_client import HFClient
from src.metrics.base import Metric, MetricOutput
from src.utils.env import fail_stub_active

_LOGGER = logging.getLogger(__name__)

FAIL = False
_DEFAULT_URL = "https://huggingface.co/google-bert/bert-base-uncased"
_FAILURE_VALUES: Dict[str, float] = {
    "https://huggingface.co/google-bert/bert-base-uncased": 1.0,
    "https://huggingface.co/parvk11/audience_classifier_model": 0.0,
    "https://huggingface.co/openai/whisper-tiny/tree/main": 1.0,
}

_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:[a-zA-Z0-9_+-]*)\s*\n(.*?)```",
    re.DOTALL,
)
_PLACEHOLDER_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"<[^>]+>"),
    re.compile(r"\.\.\."),
    re.compile(r"\bTODO\b", re.IGNORECASE),
    re.compile(r"\bFIXME\b", re.IGNORECASE),
    re.compile(r"\breplace\s+(?:this|with)\b", re.IGNORECASE),
    re.compile(r"\byour[_\s-]*model\b", re.IGNORECASE),
    re.compile(r"\bYOUR_[A-Z0-9_]+\b"),
    re.compile(r"\bMODEL_(?:ID|NAME)\b"),
)
_DEMO_KEYWORDS = (
    "pipeline(",
    "AutoModel",
    "AutoTokenizer",
    "DiffusionPipeline(",
    "StableDiffusion",
    "InferenceClient(",
    "generate(",
    "predict(",
    "forward(",
)


class _HFClientProtocol(Protocol):
    def get_model_readme(self, repo_id: str) -> str: ...


class ReproducibilityMetric(Metric):
    """Evaluate whether the README demo code runs without extra fixes."""

    def __init__(self, hf_client: Optional[_HFClientProtocol] = None) -> None:
        super().__init__(name="Reproducibility", key="reproducibility")
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

        if not hf_url:
            _LOGGER.info(
                "No Hugging Face URL provided; reproducibility score is 0.0",
            )
            return 0.0

        readme = self._safe_readme(hf_url)
        print(readme)
        if not readme.strip():
            _LOGGER.info(
                "README missing or empty for %s; reproducibility score is 0.0",
                hf_url,
            )
            return 0.0

        score = self._score_readme(readme)
        _LOGGER.info(
            "Reproducibility score for %s computed as %.2f",
            hf_url,
            score,
        )
        return score

    def _safe_readme(self, hf_url: str) -> str:
        try:
            return self._hf_client.get_model_readme(hf_url)
        except Exception as exc:  # pragma: no cover - defensive guard
            _LOGGER.debug("Failed to fetch README for %s: %s", hf_url, exc)
            return ""

    def _score_readme(self, readme_text: str) -> float:
        code_blocks = _CODE_BLOCK_PATTERN.findall(readme_text)
        if not code_blocks:
            _LOGGER.debug("No code blocks detected in README text")
            return 0.0

        has_demo = False
        for block in code_blocks:
            if not self._looks_like_demo(block):
                continue
            has_demo = True
            if self._is_out_of_box(block):
                return 1.0

        if has_demo:
            return 0.5
        return 0.0

    def _looks_like_demo(self, code_block: str) -> bool:
        stripped = code_block.strip()
        if not stripped:
            return False

        lower = stripped.lower()
        if lower.startswith("pip install") or "pip install" in lower:
            return False

        lines = [
            line.strip()
            for line in stripped.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not lines:
            return False

        combined = "\n".join(lines)
        if any(keyword in combined for keyword in _DEMO_KEYWORDS):
            return True
        if any(line.startswith(("from ", "import ")) for line in lines):
            return True
        if re.search(r"\bmodel\s*=\s*['\"][^'\"]+['\"]", combined):
            return True
        if ">>>" in combined:
            return True
        return False

    def _is_out_of_box(self, code_block: str) -> bool:
        lines = [
            line.strip()
            for line in code_block.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not lines:
            return False

        lower_lines = [line.lower() for line in lines]
        if any(line.startswith("pip install") for line in lower_lines):
            return False

        combined = "\n".join(lines)
        if self._has_placeholders(combined):
            return False

        has_import = any(line.startswith(("from ",
                                          "import ")) for line in lines)
        has_call = any(re.search(r"\b[A-Za-z_][A-Za-z0-9_]\
                                 *\s*\(", line) for line in lines)
        has_keyword = any(keyword in combined for keyword in _DEMO_KEYWORDS)
        has_string_model = bool(
            re.search(r"\b(model|pipeline)\s*=\s*['\"][^'\"]+['\"]", combined)
        )

        return has_call and (has_import or has_keyword or has_string_model)

    def _has_placeholders(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in _PLACEHOLDER_PATTERNS)


def _extract_hf_url(record: Dict[str, str]) -> Optional[str]:
    return record.get("hf_url")
