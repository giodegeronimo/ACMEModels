"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Code quality metric assessing repository hygiene and signals.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from src.clients.git_client import GitClient
from src.clients.hf_client import HFClient
from src.metrics.base import Metric, MetricOutput
from src.utils.env import enable_readme_fallback, fail_stub_active

_LOGGER = logging.getLogger(__name__)

FAIL = False
_DEFAULT_URL = "https://huggingface.co/google-bert/bert-base-uncased"

_FAILURE_VALUES: Dict[str, float] = {
    "https://huggingface.co/google-bert/bert-base-uncased": 0.2,
    "https://huggingface.co/parvk11/audience_classifier_model": 0.8,
    "https://huggingface.co/openai/whisper-tiny/tree/main": 0.8,
}

CORE_HYGIENE_WEIGHT = 0.6
DOCS_WEIGHT = 0.3
ACTIVITY_MAX_BONUS = 0.1

_GITHUB_URL_PATTERN = re.compile(
    r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)"
)

_CI_PATTERNS = (
    ".github/workflows/",
    ".travis.yml",
    "azure-pipelines",
    ".circleci/config",
    "jenkinsfile",
)
_LINT_FILES = (
    ".pre-commit-config.yaml",
    ".flake8",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    ".editorconfig",
)
_TYPE_FILES = (
    "mypy.ini",
    "pyproject.toml",
    "pyrightconfig.json",
    "setup.cfg",
)


class _HFClientProtocol(Protocol):
    """
    get_model_readme: Function description.
    :param repo_id:
    :returns:
    """

    """
    _HFClientProtocol: Class description.
    """

    def get_model_readme(self, repo_id: str) -> str: ...


class _GitClientProtocol(Protocol):
    """
    get_repo_metadata: Function description.
    :param repo_url:
    :returns:
    """

    """
    _GitClientProtocol: Class description.
    """

    def get_repo_metadata(self, repo_url: str) -> Dict[str, Any]: ...

    def list_repo_files(
        self, repo_url: str, *, branch: Optional[str] = None
    ) -> List[str]: ...


class CodeQualityMetric(Metric):
    """Estimate repository hygiene and documentation quality."""

    def __init__(
        self,
        hf_client: Optional[_HFClientProtocol] = None,
        git_client: Optional[_GitClientProtocol] = None,
    ) -> None:
        """
        __init__: Function description.
        :param hf_client:
        :param git_client:
        :returns:
        """

        super().__init__(name="Code Quality", key="code_quality")
        self._hf = hf_client or HFClient()
        self._git = git_client or GitClient()

    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        """
        compute: Function description.
        :param url_record:
        :returns:
        """

        if fail_stub_active(FAIL):
            time.sleep(0.05)
            url = _extract_hf_url(url_record) or _DEFAULT_URL
            return _FAILURE_VALUES.get(url, _FAILURE_VALUES[_DEFAULT_URL])

        hf_url = _extract_hf_url(url_record)
        if not hf_url:
            return 0.0

        readme_text = self._safe_readme(hf_url)
        repo_url = self._select_repo_url(url_record, readme_text)

        repo_metadata: Optional[Dict[str, Any]] = None
        repo_files: List[str] = []
        if repo_url:
            repo_metadata = self._safe_repo_metadata(repo_url)
            repo_files = self._safe_repo_files(repo_url, repo_metadata)

        hygiene_score, hygiene_details = self._hygiene_score(repo_files)
        docs_score, docs_details = self._docs_score(readme_text, repo_files)
        base_score = (
            CORE_HYGIENE_WEIGHT * hygiene_score
            + DOCS_WEIGHT * docs_score
        )

        activity_bonus, activity_details = self._activity_bonus(repo_metadata)
        final_score = min(1.0, base_score + activity_bonus)

        _LOGGER.info(
            "Code quality metrics for %s: hygiene=%.2f details=%s docs=%.2f "
            "details=%s activity_bonus=%.2f details=%s",
            hf_url,
            hygiene_score,
            hygiene_details,
            docs_score,
            docs_details,
            activity_bonus,
            activity_details,
        )

        return final_score

    def _select_repo_url(
        self,
        url_record: Dict[str, str],
        readme_text: str,
    ) -> Optional[str]:
        """
        _select_repo_url: Function description.
        :param url_record:
        :param readme_text:
        :returns:
        """

        git_url = (url_record.get("git_url") or "").strip()
        if git_url:
            return _normalize_github_url(git_url)

        if not enable_readme_fallback():
            _LOGGER.info(
                "Code quality: README repo fallback disabled for %s",
                url_record.get("hf_url"),
            )
            return None

        for match in _GITHUB_URL_PATTERN.finditer(readme_text or ""):
            candidate = f"https://github.com/{match.group(1)}/{match.group(2)}"
            normalized = _normalize_github_url(candidate)
            if normalized:
                return normalized
        return None

    def _safe_readme(self, hf_url: str) -> str:
        """
        _safe_readme: Function description.
        :param hf_url:
        :returns:
        """

        try:
            return self._hf.get_model_readme(hf_url) or ""
        except Exception as exc:
            _LOGGER.debug("Failed to fetch README for %s: %s", hf_url, exc)
            return ""

    def _safe_repo_metadata(self, repo_url: str) -> Optional[Dict[str, Any]]:
        """
        _safe_repo_metadata: Function description.
        :param repo_url:
        :returns:
        """

        try:
            return self._git.get_repo_metadata(repo_url)
        except Exception as exc:
            _LOGGER.info(
                "Repository metadata unavailable for %s: %s",
                repo_url,
                exc,
            )
            return None

    def _safe_repo_files(
        self,
        repo_url: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[str]:
        """
        _safe_repo_files: Function description.
        :param repo_url:
        :param metadata:
        :returns:
        """

        branch = None
        if metadata:
            branch = metadata.get("default_branch") or metadata.get(
                "main_branch"
            )
        try:
            return self._git.list_repo_files(repo_url, branch=branch)
        except Exception as exc:
            _LOGGER.info(
                "Repository tree unavailable for %s: %s",
                repo_url,
                exc,
            )
            return []

    def _hygiene_score(
        self, repo_files: Sequence[str]
    ) -> tuple[float, Dict[str, float]]:
        """
        _hygiene_score: Function description.
        :param repo_files:
        :returns:
        """

        repo_files = list(repo_files)
        tests_present = any(
            _is_test_path(path) for path in repo_files
        )
        ci_present = any(
            _matches_any(path, _CI_PATTERNS) for path in repo_files
        )
        lint_present = any(
            _matches_any(path, _LINT_FILES) for path in repo_files
        )
        type_present = any(
            _matches_any(path, _TYPE_FILES) for path in repo_files
        )

        code_files = [path for path in repo_files if path.endswith(".py")]
        code_ratio = 0.0
        if repo_files:
            code_ratio = min(1.0, len(code_files) / len(repo_files))

        signals = [
            1.0 if tests_present else 0.0,
            1.0 if ci_present else 0.0,
            1.0 if lint_present else 0.0,
            1.0 if type_present else 0.0,
            code_ratio,
        ]
        score = sum(signals) / len(signals)
        details = {
            "tests": float(tests_present),
            "ci": float(ci_present),
            "lint": float(lint_present),
            "types": float(type_present),
            "code_ratio": round(code_ratio, 2),
        }
        return score, details

    def _docs_score(
        self, readme_text: str, repo_files: Sequence[str]
    ) -> tuple[float, Dict[str, float]]:
        """
        _docs_score: Function description.
        :param readme_text:
        :param repo_files:
        :returns:
        """

        signals: List[float] = []
        detail_flags: Dict[str, float] = {}
        if readme_text:
            headings = {
                heading.strip().lower()
                for heading in re.findall(
                    r"^##+\s+(.+)$",
                    readme_text,
                    re.MULTILINE,
                )
            }
            if any("install" in heading for heading in headings):
                signals.append(1.0)
                detail_flags["install_section"] = 1.0
            if any(
                "usage" in heading or "quickstart" in heading
                for heading in headings
            ):
                signals.append(1.0)
                detail_flags["usage_section"] = 1.0
            if any("test" in heading for heading in headings):
                signals.append(1.0)
                detail_flags["test_section"] = 1.0
            if any("contribut" in heading for heading in headings):
                signals.append(1.0)
                detail_flags["contrib_section"] = 1.0
            length_bonus = min(1.0, len(readme_text) / 1000.0)
            signals.append(length_bonus)
            detail_flags["readme_length"] = round(length_bonus, 2)

        repo_lower = [path.lower() for path in repo_files]
        if any(path.startswith("docs/") for path in repo_lower):
            signals.append(1.0)
            detail_flags["docs_folder"] = 1.0
        if any(
            path.endswith("contributing.md") for path in repo_lower
        ):
            signals.append(1.0)
            detail_flags["contrib_file"] = 1.0

        if not signals:
            return 0.0, detail_flags
        score = sum(signals) / len(signals)
        return score, detail_flags

    def _activity_bonus(
        self, metadata: Optional[Dict[str, Any]]
    ) -> tuple[float, Dict[str, Any]]:
        """
        _activity_bonus: Function description.
        :param metadata:
        :returns:
        """

        if not metadata:
            return 0.0, {"reason": "no_metadata"}

        bonus = 0.0
        is_archived = bool(metadata.get("archived"))
        if not is_archived:
            timestamp = metadata.get("pushed_at") or metadata.get("updated_at")
            if isinstance(timestamp, str):
                days = _days_since(timestamp)
                if days is not None:
                    if days <= 90:
                        bonus += 0.07
                    elif days <= 180:
                        bonus += 0.04

        stars = metadata.get("stargazers_count") or metadata.get(
            "watchers_count"
        )
        forks = metadata.get("forks_count")
        popularity = 0
        if isinstance(stars, int):
            popularity = stars
        elif isinstance(forks, int):
            popularity = forks

        if popularity >= 500:
            bonus += 0.05
        elif popularity >= 100:
            bonus += 0.03
        elif popularity >= 10:
            bonus += 0.02

        capped = min(ACTIVITY_MAX_BONUS, bonus)
        details = {
            "archived": float(is_archived),
            "raw_bonus": round(bonus, 3),
            "capped_bonus": round(capped, 3),
            "popularity": popularity,
        }
        return capped, details


def _extract_hf_url(record: Dict[str, str]) -> Optional[str]:
    """
    _extract_hf_url: Function description.
    :param record:
    :returns:
    """

    return record.get("hf_url")


def _normalize_github_url(url: str) -> Optional[str]:
    """
    _normalize_github_url: Function description.
    :param url:
    :returns:
    """

    match = _GITHUB_URL_PATTERN.search(url)
    if not match:
        return None
    owner, repo = match.group(1), match.group(2)
    return f"https://github.com/{owner}/{repo}"


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    """
    _matches_any: Function description.
    :param path:
    :param patterns:
    :returns:
    """

    lower = path.lower()
    return any(pattern.lower() in lower for pattern in patterns)


def _is_test_path(path: str) -> bool:
    """
    _is_test_path: Function description.
    :param path:
    :returns:
    """

    lower = path.lower()
    return (
        lower.startswith("tests/")
        or lower.startswith("test/")
        or lower.endswith("_test.py")
        or lower.endswith("test.py")
    )


def _days_since(timestamp: str) -> Optional[float]:
    """
    _days_since: Function description.
    :param timestamp:
    :returns:
    """

    try:
        parsed = _parse_iso_datetime(timestamp)
    except ValueError:
        return None
    now = datetime.now(timezone.utc)
    delta = now - parsed
    return delta.days + (delta.seconds / 86400.0)


def _parse_iso_datetime(timestamp: str) -> datetime:
    """
    _parse_iso_datetime: Function description.
    :param timestamp:
    :returns:
    """

    cleaned = timestamp.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    return datetime.fromisoformat(cleaned)
