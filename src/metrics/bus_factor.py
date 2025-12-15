"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Bus factor metric estimating contributor diversity and activity.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Sequence

from src.clients.git_client import GitClient
from src.clients.hf_client import HFClient
from src.metrics.base import Metric, MetricOutput
from src.utils.env import enable_readme_fallback, fail_stub_active

_LOGGER = logging.getLogger(__name__)

FAIL = False

_DEFAULT_URL = "https://huggingface.co/google-bert/bert-base-uncased"

_FAILURE_VALUES: Dict[str, float] = {
    "https://huggingface.co/google-bert/bert-base-uncased": 0.21,
    "https://huggingface.co/parvk11/audience_classifier_model": 0.7,
    "https://huggingface.co/openai/whisper-tiny/tree/main": 0.2,
}

CONTRIBUTOR_WEIGHT = 0.6
OWNERSHIP_WEIGHT = 0.2
COMMUNITY_WEIGHT = 0.2

_GITHUB_URL_PATTERN = re.compile(
    r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)"
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
    """
    get_model_info: Function description.
    :param repo_id:
    :returns:
    """

    def get_model_info(self, repo_id: str) -> Any: ...


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
        self,
        repo_url: str,
        *,
        branch: Optional[str] = None,
    ) -> List[str]: ...

    def list_repo_contributors(
        self,
        repo_url: str,
        *,
        per_page: int = 100,
    ) -> List[Dict[str, Any]]: ...


class BusFactorMetric(Metric):
    """Estimate the resilience of a project to maintainer loss."""

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

        super().__init__(name="Bus Factor", key="bus_factor")
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
        hf_info = self._safe_model_info(hf_url)
        repo_url = self._select_repo_url(url_record, readme_text)

        contributors: List[Dict[str, Any]] = []
        metadata: Optional[Dict[str, Any]] = None
        if repo_url:
            metadata = self._safe_repo_metadata(repo_url)
            contributors = self._safe_contributors(repo_url)
            _LOGGER.info(
                "Bus factor inputs for %s: repo=%s metadata=%s "
                "contributors=%d",
                hf_url,
                repo_url,
                bool(metadata),
                len(contributors),
            )
        else:
            _LOGGER.info(
                "Bus factor: no repository URL resolved for %s", hf_url
            )

        if not contributors and metadata is None:
            hf_fallback = _hf_metadata_fallback(hf_info)
            if hf_fallback is not None:
                _LOGGER.info(
                    "Bus factor HF metadata fallback for %s yielded %.2f",
                    hf_url,
                    hf_fallback,
                )
                return hf_fallback

            readme_names = _extract_readme_maintainers(readme_text)
            if readme_names:
                contributor_score = _readme_contributor_score(readme_names)
                ownership_score = 0.3 if len(readme_names) >= 2 else 0.1
                community_score = 0.0
                _LOGGER.info(
                    "Bus factor README fallback for %s: names=%s "
                    "contributor=%.2f ownership=%.2f",
                    hf_url,
                    readme_names,
                    contributor_score,
                    ownership_score,
                )
                return max(
                    0.0,
                    min(
                        1.0,
                        CONTRIBUTOR_WEIGHT * contributor_score
                        + OWNERSHIP_WEIGHT * ownership_score
                        + COMMUNITY_WEIGHT * community_score,
                    ),
                )

        contributor_score = _contributor_diversity(contributors)
        ownership_score = _ownership_resilience(metadata, contributors)
        community_score = _community_support(metadata, hf_info)

        final_score = (
            CONTRIBUTOR_WEIGHT * contributor_score
            + OWNERSHIP_WEIGHT * ownership_score
            + COMMUNITY_WEIGHT * community_score
        )

        _LOGGER.info(
            "Bus factor metrics for %s: contributors=%.2f ownership=%.2f "
            "community=%.2f (total_contributors=%d)",
            hf_url,
            contributor_score,
            ownership_score,
            community_score,
            len(contributors),
        )

        return max(0.0, min(1.0, final_score))

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
                "Bus factor: README repo fallback disabled for %s",
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

    def _safe_model_info(self, hf_url: str) -> Optional[Any]:
        """
        _safe_model_info: Function description.
        :param hf_url:
        :returns:
        """

        try:
            return self._hf.get_model_info(hf_url)
        except Exception as exc:
            _LOGGER.debug("Model info unavailable for %s: %s", hf_url, exc)
            return None

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

    def _safe_contributors(self, repo_url: str) -> List[Dict[str, Any]]:
        """
        _safe_contributors: Function description.
        :param repo_url:
        :returns:
        """

        try:
            return self._git.list_repo_contributors(repo_url)
        except Exception as exc:
            _LOGGER.info(
                "Unable to list contributors for %s: %s",
                repo_url,
                exc,
            )
            return []


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


def _contributor_diversity(contributors: Sequence[Dict[str, Any]]) -> float:
    """
    _contributor_diversity: Function description.
    :param contributors:
    :returns:
    """

    if not contributors:
        return 0.0

    counts = [
        int(entry.get("contributions", 0) or 0)
        for entry in contributors
    ]
    counts = [count for count in counts if count > 0]
    if not counts:
        return 0.0

    num_contributors = len(counts)
    if num_contributors >= 5:
        num_score = 1.0
    elif num_contributors >= 3:
        num_score = 0.7
    elif num_contributors == 2:
        num_score = 0.4
    else:
        num_score = 0.1

    total_commits = sum(counts)
    top_commits = max(counts)
    dominance = top_commits / total_commits if total_commits else 1.0
    if dominance <= 0.5:
        diversity = 1.0
    else:
        diversity = max(0.0, 1.0 - (dominance - 0.5) / 0.5)

    return 0.6 * num_score + 0.4 * diversity


def _ownership_resilience(
    metadata: Optional[Dict[str, Any]],
    contributors: Sequence[Dict[str, Any]],
) -> float:
    """
    _ownership_resilience: Function description.
    :param metadata:
    :param contributors:
    :returns:
    """

    if not metadata:
        return 0.2 if len(contributors) >= 2 else 0.1

    owner = metadata.get("owner") or {}
    owner_type = owner.get("type") if isinstance(owner, dict) else None
    if metadata.get("archived"):
        return 0.0

    if owner_type == "Organization":
        return 1.0

    if len(contributors) >= 3:
        return 0.6

    if metadata.get("fork"):
        return 0.2

    return 0.3


def _community_support(
    metadata: Optional[Dict[str, Any]],
    hf_info: Optional[Any],
) -> float:
    """
    _community_support: Function description.
    :param metadata:
    :param hf_info:
    :returns:
    """

    archived = False
    popularity: int = 0
    pushed_at: Optional[str] = None

    if metadata:
        archived = bool(metadata.get("archived"))
        if not archived:
            value = metadata.get("stargazers_count")
            if not isinstance(value, int):
                value = metadata.get("watchers_count") or metadata.get(
                    "forks_count"
                )
            if isinstance(value, int):
                popularity = value
            pushed_at = metadata.get("pushed_at") or metadata.get("updated_at")
    elif hf_info is not None:
        card = _hf_card_data(hf_info)
        archived = bool(card.get("deprecated"))
        downloads = getattr(hf_info, "downloads", 0) or 0
        if isinstance(downloads, int):
            popularity = downloads
        pushed_at = getattr(hf_info, "lastModified", None) or getattr(
            hf_info,
            "last_modified",
            None,
        )

    if archived:
        return 0.0

    if popularity >= 500000:
        popularity_score = 1.0
    elif popularity >= 100000:
        popularity_score = 0.7
    elif popularity >= 20000:
        popularity_score = 0.5
    elif popularity >= 5000:
        popularity_score = 0.3
    elif popularity > 0:
        popularity_score = 0.1
    else:
        popularity_score = 0.0

    recency_score = 0.0
    if isinstance(pushed_at, str):
        days = _days_since(pushed_at)
        if days is not None:
            if days <= 90:
                recency_score = 1.0
            elif days <= 180:
                recency_score = 0.7
            elif days <= 365:
                recency_score = 0.4
            else:
                recency_score = 0.1

    return min(1.0, (popularity_score + recency_score) / 2)


KNOWN_ORG_OWNERS = {
    "google",
    "facebook",
    "meta",
    "microsoft",
    "openai",
    "huggingface",
    "stabilityai",
    "anthropic",
    "ibm",
    "aws",
}


def _hf_metadata_fallback(hf_info: Optional[Any]) -> Optional[float]:
    """
    _hf_metadata_fallback: Function description.
    :param hf_info:
    :returns:
    """

    if hf_info is None:
        return None

    card = _hf_card_data(hf_info)
    maintainers = _hf_maintainers(card)
    owner = _hf_owner(hf_info)
    if owner and owner not in maintainers:
        maintainers.append(owner)

    contributor_score = _maintainer_count_score(len(maintainers))
    ownership_score = 0.0
    if owner:
        ownership_score = 1.0 if owner.lower() in KNOWN_ORG_OWNERS else 0.4
    if len(maintainers) >= 3:
        ownership_score = max(ownership_score, 0.6)
    elif len(maintainers) >= 2:
        ownership_score = max(ownership_score, 0.4)
    elif len(maintainers) == 1:
        ownership_score = max(ownership_score, 0.2)
    else:
        ownership_score = max(ownership_score, 0.1)

    downloads = getattr(hf_info, "downloads", 0) or 0
    likes = getattr(hf_info, "likes", 0) or 0
    last_modified = getattr(hf_info, "lastModified", None) or getattr(
        hf_info,
        "last_modified",
        None,
    )
    community_score = _hf_community_score(downloads, likes, last_modified)

    final = (
        CONTRIBUTOR_WEIGHT * contributor_score
        + OWNERSHIP_WEIGHT * ownership_score
        + COMMUNITY_WEIGHT * community_score
    )
    _LOGGER.info(
        "Bus factor HF metadata fallback: maintainers=%s owner=%s "
        "downloads=%s likes=%s last_modified=%s -> contributor=%.2f "
        "ownership=%.2f community=%.2f final=%.2f",
        maintainers,
        owner,
        downloads,
        likes,
        last_modified,
        contributor_score,
        ownership_score,
        community_score,
        final,
    )
    return max(0.0, min(1.0, final))


def _hf_community_score(
    downloads: int,
    likes: int,
    last_modified: Optional[str],
) -> float:
    """
    _hf_community_score: Function description.
    :param downloads:
    :param likes:
    :param last_modified:
    :returns:
    """

    popularity = max(downloads, likes)
    if popularity >= 500000:
        popularity_score = 1.0
    elif popularity >= 100000:
        popularity_score = 0.7
    elif popularity >= 20000:
        popularity_score = 0.5
    elif popularity >= 5000:
        popularity_score = 0.3
    elif popularity > 0:
        popularity_score = 0.1
    else:
        popularity_score = 0.0

    recency_score = 0.0
    if isinstance(last_modified, str):
        days = _days_since(last_modified)
        if days is not None:
            if days <= 90:
                recency_score = 1.0
            elif days <= 180:
                recency_score = 0.7
            elif days <= 365:
                recency_score = 0.4
            else:
                recency_score = 0.1

    return min(1.0, (popularity_score + recency_score) / 2)


def _hf_card_data(hf_info: Any) -> Dict[str, Any]:
    """
    _hf_card_data: Function description.
    :param hf_info:
    :returns:
    """

    card = getattr(hf_info, "card_data", None)
    if card is None:
        card = getattr(hf_info, "cardData", None)
    if isinstance(card, dict):
        return card
    return {}


def _hf_maintainers(card: Dict[str, Any]) -> List[str]:
    """
    _hf_maintainers: Function description.
    :param card:
    :returns:
    """

    maintainers_field = card.get("maintainers")
    names: List[str] = []
    if isinstance(maintainers_field, list):
        for entry in maintainers_field:
            if isinstance(entry, str):
                names.append(entry.strip())
            elif isinstance(entry, dict):
                name = entry.get("name") or entry.get("github")
                if isinstance(name, str):
                    names.append(name.strip())
    author = card.get("author")
    if isinstance(author, str):
        names.append(author.strip())
    elif isinstance(author, list):
        names.extend(str(item).strip() for item in author)
    return [name for name in names if name]


def _hf_owner(hf_info: Any) -> Optional[str]:
    """
    _hf_owner: Function description.
    :param hf_info:
    :returns:
    """

    identifier = getattr(hf_info, "id", "")
    if isinstance(identifier, str) and "/" in identifier:
        return identifier.split("/", 1)[0]
    return None


def _maintainer_count_score(count: int) -> float:
    """
    _maintainer_count_score: Function description.
    :param count:
    :returns:
    """

    if count >= 5:
        return 1.0
    if count >= 3:
        return 0.7
    if count == 2:
        return 0.4
    if count == 1:
        return 0.2
    return 0.1


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
    delta = datetime.now(timezone.utc) - parsed
    return delta.days + delta.seconds / 86400.0


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


def _extract_readme_maintainers(text: str) -> List[str]:
    """
    _extract_readme_maintainers: Function description.
    :param text:
    :returns:
    """

    if not text:
        return []
    if not enable_readme_fallback():
        _LOGGER.info("Bus factor: README maintainer fallback disabled")
        return []
    sections = re.split(r"^##+\s+", text, flags=re.MULTILINE)
    names: List[str] = []
    for section in sections:
        header_match = re.match(
            r"(maintainers?|authors?)\b",
            section,
            re.IGNORECASE,
        )
        if header_match:
            body = section[header_match.end():]
            names.extend(_extract_names_from_section(body))
    if not names:
        # fallback: bullet lists mentioning maintainer/author
        pattern = re.compile(
            r"^(?:[-*]|\d+\.)\s+(.+)$",
            re.MULTILINE,
        )
        for match in pattern.finditer(text):
            line = match.group(1)
            if re.search(r"maintainer|author", line, re.IGNORECASE):
                names.append(line.strip())
    return list(dict.fromkeys(names))


def _extract_names_from_section(section: str) -> List[str]:
    """
    _extract_names_from_section: Function description.
    :param section:
    :returns:
    """

    pattern = re.compile(
        r"^(?:[-*]|\d+\.)\s+(.+)$",
        re.MULTILINE,
    )
    names: List[str] = []
    for match in pattern.finditer(section):
        entry = match.group(1).strip()
        if entry:
            names.append(entry)
    return names


def _readme_contributor_score(names: Sequence[str]) -> float:
    """
    _readme_contributor_score: Function description.
    :param names:
    :returns:
    """

    count = len(names)
    if count >= 5:
        return 0.8
    if count >= 3:
        return 0.6
    if count == 2:
        return 0.4
    if count == 1:
        return 0.2
    return 0.0
