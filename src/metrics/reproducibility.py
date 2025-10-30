from __future__ import annotations

import contextlib
import logging
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Pattern, Protocol, Sequence

from src.clients.hf_client import HFClient
from src.config import LLM_ANALYSIS_MODEL, LLM_TEMPERATURE
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

_EXECUTION_TIMEOUT_SECONDS = 20
_LLM_MAX_ATTEMPTS = 3
_LLM_FIX_SYSTEM_PROMPT = (
    "You repair short Python scripts copied from README files so they run "
    "successfully without manual edits. Always return the full corrected "
    "Python script and nothing else."
)
_LLM_INITIAL_USER_PROMPT = (
    "Here is Python code from a README and the error it produced when "
    "executed.\n\n"
    "Code:\n```python\n{code}\n```\n\n"
    "Error output:\n{error}\n\n"
    "Provide a corrected Python script that will run successfully. "
    "Return only the code."
)
_LLM_RETRY_PROMPT = (
    "That attempt still failed. The latest error output was:\n{error}\n\n"
    "Please try again and return only the updated Python script."
)


@dataclass
class _CodeBlock:
    language: str
    code: str


class _PurdueClientProtocol(Protocol):
    def llm(
        self,
        prompt: Optional[str] = None,
        *,
        messages: Optional[Sequence[Dict[str, str]]] = None,
        model: str = ...,
        stream: bool = ...,
        temperature: float = ...,
        **extra: object,
    ) -> str: ...


_CODE_BLOCK_PATTERN = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_+-]*)\s*\n(?P<code>.*?)```",
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

    def __init__(
        self,
        hf_client: Optional[_HFClientProtocol] = None,
        purdue_client: Optional[_PurdueClientProtocol] = None,
    ) -> None:
        super().__init__(name="Reproducibility", key="reproducibility")
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

        if not hf_url:
            _LOGGER.info(
                "No Hugging Face URL provided; reproducibility score is 0.0",
            )
            return 0.0

        readme = self._safe_readme(hf_url)
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

    def _extract_code_blocks(self, readme_text: str) -> list[_CodeBlock]:
        blocks: list[_CodeBlock] = []
        for match in _CODE_BLOCK_PATTERN.finditer(readme_text):
            language = (match.group("lang") or "").strip().lower()
            code = match.group("code") or ""
            if code.strip():
                blocks.append(_CodeBlock(language=language, code=code))
        return blocks

    def _run_code_block(self, code_block: str) -> tuple[bool, str]:
        dedented = textwrap.dedent(code_block).strip()
        if not dedented:
            return False, "Code block is empty after dedent"

        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                suffix=".py",
                delete=False,
                encoding="utf-8",
            ) as handle:
                tmp_path = Path(handle.name)
                handle.write(dedented)

            completed = subprocess.run(
                [sys.executable, str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=_EXECUTION_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return False, (
                f"Execution timed out after {_EXECUTION_TIMEOUT_SECONDS} "
                "seconds"
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            return False, f"Execution failed: {exc}"
        finally:
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    tmp_path.unlink(missing_ok=True)

        if completed.returncode == 0:
            return True, ""

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if stdout and stderr:
            error_output = f"{stdout}\n{stderr}"
        else:
            error_output = stdout or stderr or (
                f"Process exited with status {completed.returncode}"
            )
        return False, error_output

    def _attempt_llm_fix(self, code_block: str, error_output: str) -> bool:
        client = self._get_purdue_client()
        if client is None:
            _LOGGER.debug("Purdue client unavailable; skipping LLM fixes")
            return False

        error_text = error_output or "Process failed without error output."
        messages: list[Dict[str, str]] = [
            {"role": "system", "content": _LLM_FIX_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _LLM_INITIAL_USER_PROMPT.format(
                    code=textwrap.dedent(code_block).strip(),
                    error=error_text,
                ),
            },
        ]

        current_error = error_text
        for attempt in range(1, _LLM_MAX_ATTEMPTS + 1):
            try:
                response = client.llm(
                    messages=list(messages),
                    model=LLM_ANALYSIS_MODEL,
                    temperature=LLM_TEMPERATURE,
                )
            except Exception as exc:  # pragma: no cover - external service
                _LOGGER.debug(
                    "LLM fix attempt %d failed to complete: %s",
                    attempt,
                    exc,
                )
                return False

            messages.append({"role": "assistant", "content": response})
            candidate = self._extract_candidate_code(response)
            if not candidate:
                _LOGGER.debug(
                    "LLM fix attempt %d returned no code; retrying", attempt
                )
                messages.append(
                    {
                        "role": "user",
                        "content": _LLM_RETRY_PROMPT.format(error=current_error),
                    }
                )
                continue

            success, current_error = self._run_code_block(candidate)
            if success:
                _LOGGER.info(
                    "LLM fix succeeded on attempt %d", attempt
                )
                return True

            messages.append(
                {
                    "role": "user",
                    "content": _LLM_RETRY_PROMPT.format(error=current_error),
                }
            )

        _LOGGER.debug("LLM unable to repair code after %d attempts", _LLM_MAX_ATTEMPTS)
        return False

    def _extract_candidate_code(self, llm_response: str) -> str:
        match = re.search(
            r"```(?:python)?\s*\n(?P<code>.*?)```",
            llm_response,
            re.DOTALL,
        )
        if match:
            return match.group("code").strip()
        return llm_response.strip()

    def _get_purdue_client(self) -> Optional[_PurdueClientProtocol]:
        if self._purdue_client is not None:
            return self._purdue_client

        try:
            from src.clients.purdue_client import PurdueClient

            self._purdue_client = PurdueClient()
        except Exception as exc:
            _LOGGER.info(
                "Purdue client unavailable for reproducibility fixes: %s",
                exc,
            )
            self._purdue_client = None
        return self._purdue_client

    def _safe_readme(self, hf_url: str) -> str:
        try:
            return self._hf_client.get_model_readme(hf_url)
        except Exception as exc:  # pragma: no cover - defensive guard
            _LOGGER.debug("Failed to fetch README for %s: %s", hf_url, exc)
            return ""

    def _score_readme(self, readme_text: str) -> float:
        blocks = self._extract_code_blocks(readme_text)
        if not blocks:
            _LOGGER.debug("No code blocks detected in README text")
            return 0.0

        demo_blocks: list[_CodeBlock] = [
            block
            for block in blocks
            if self._looks_like_demo(block.code)
            and block.language in {"", "python", "py"}
        ]

        if not demo_blocks:
            _LOGGER.debug("No runnable demo blocks detected in README")
            return 0.0

        for block in demo_blocks:
            success, error_output = self._run_code_block(block.code)
            if success:
                return 1.0
            if self._has_placeholders(block.code):
                _LOGGER.debug("Skipping LLM fix due to placeholders in block")
                continue
            if self._attempt_llm_fix(block.code, error_output):
                return 0.5

        _LOGGER.debug("All demo blocks failed to run after LLM attempts")
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

    def _has_placeholders(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in _PLACEHOLDER_PATTERNS)


def _extract_hf_url(record: Dict[str, str]) -> Optional[str]:
    return record.get("hf_url")
