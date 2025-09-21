''' A module to classify URLs (HuggingFace, GitHub, etc.), fetch metadata, and read README files.

    This is a chatgpt generated base code and is being modified to fit the requirements.
'''



from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

try:
    import requests
except ImportError as e:
    raise SystemExit(
        "The 'requests' package is required by url_fetcher.py. "
        "Please run './run install' to install dependencies."
    ) from e


# ------------------------- Logging ------------------------- #

def _make_logger() -> logging.Logger:
    logger = logging.getLogger("url_fetcher")
    if getattr(logger, "_configured", False):
        return logger

    levelEnv = os.environ.get("LOG_LEVEL", "0").strip()
    try:
        levelNum = int(levelEnv)
    except ValueError:
        levelNum = 0

    if levelNum <= 0:
        level = logging.CRITICAL + 1  # effectively silent
    elif levelNum == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logger.setLevel(level)
    logger.propagate = False

    handler: logging.Handler
    logFile = os.environ.get("LOG_FILE")
    handler = logging.FileHandler(logFile) if logFile else logging.NullHandler()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | url_fetcher | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    setattr(logger, "_configured", True)
    return logger


LOG = _make_logger()


# ------------------------- Types ------------------------- #

class UrlCategory(str, Enum):
    MODEL = "MODEL"
    DATASET = "DATASET"
    CODE = "CODE"
    UNKNOWN = "UNKNOWN"


class Host(str, Enum):
    HUGGINGFACE = "huggingface"
    GITHUB = "github"
    OTHER = "other"


@dataclass(frozen=True)
class ResourceRef:
    url: str
    host: Host
    category: UrlCategory
    owner: Optional[str]
    name: Optional[str]
    repoId: Optional[str]
    normalizedUrl: str


# ------------------------- HTTP Session & Helpers ------------------------- #

DEFAULT_TIMEOUT = 10.0
MAX_RETRIES = 2
BACKOFF_SEC = 1.0

_session: Optional[requests.Session] = None
_cache: Dict[str, Any] = {}


def _get_session() -> requests.Session:
    global _session
    if _session is not None:
        return _session
    s = requests.Session()

    # Auth headers if tokens present
    ghToken = os.environ.get("GITHUB_TOKEN")
    hfToken = os.environ.get("HUGGINGFACE_TOKEN")

    s.headers.update({"User-Agent": "acme-cli-url-fetcher/1.0"})
    # We’ll add per-request Authorization headers (HF vs GH) in fetchers.
    _session = s
    return s


def _http_get_json(url: str, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
    if url in _cache:
        return _cache[url]

    sess = _get_session()
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = sess.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 429 and attempt < MAX_RETRIES:
                LOG.info(f"429 from {url}; backing off")
                time.sleep(BACKOFF_SEC * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            _cache[url] = data
            return data
        except requests.RequestException as e:
            LOG.info(f"HTTP error for {url}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_SEC * (attempt + 1))
            else:
                return None
        except json.JSONDecodeError:
            LOG.info(f"JSON decode error for {url}")
            return None
    return None


def _http_get_text(url: str, headers: Dict[str, str]) -> Optional[str]:
    cacheKey = f"text::{url}"
    if cacheKey in _cache:
        return _cache[cacheKey]
    sess = _get_session()
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = sess.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 429 and attempt < MAX_RETRIES:
                LOG.info(f"429 from {url}; backing off")
                time.sleep(BACKOFF_SEC * (attempt + 1))
                continue
            resp.raise_for_status()
            text = resp.text
            _cache[cacheKey] = text
            return text
        except requests.RequestException as e:
            LOG.info(f"HTTP error for {url}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_SEC * (attempt + 1))
            else:
                return None
    return None


def clearCache() -> None:
    """Clear the in-memory HTTP cache (plan mentions bounded batches & clearing)."""
    _cache.clear()


# ------------------------- Resource Abstraction ------------------------- #

class Resource(ABC):
    """Abstract resource per Project Plan (cast via determiner)."""

    def __init__(self, ref: ResourceRef) -> None:
        self.ref = ref

    @abstractmethod
    def fetchMetadata(self) -> Dict[str, Any]:
        """Return minimal, source-appropriate metadata (non-throwing)."""

    def fetchReadme(self) -> Optional[str]:
        """Best-effort README (optional)."""
        return None

class NoopResource(Resource):
    """Concrete, do-nothing resource so UNKNOWN hosts never raise."""
    def fetchMetadata(self) -> Dict[str, Any]:
        return {}
    def fetchReadme(self) -> Optional[str]:
        return None

class ModelResource(Resource):
    def fetchMetadata(self) -> Dict[str, Any]:
        # Hugging Face models API
        headers = {"Accept": "application/json"}
        hfToken = os.environ.get("HUGGINGFACE_TOKEN")
        if hfToken:
            headers["Authorization"] = f"Bearer {hfToken}"

        if not self.ref.repoId:
            LOG.info("Missing repoId for HF model")
            return {}

        apiUrl = f"https://huggingface.co/api/models/{self.ref.repoId}"
        data = _http_get_json(apiUrl, headers)
        meta = {}
        if data:
            meta = {
                "downloads": data.get("downloads"),
                "likes": data.get("likes"),
                "lastModified": data.get("lastModified"),
                "sha": data.get("sha"),
                "fileCount": len(data.get("siblings") or []),
            }
        return meta

    def fetchReadme(self) -> Optional[str]:
        hfToken = os.environ.get("HUGGINGFACE_TOKEN")
        headers = {"Accept": "text/plain"}
        if hfToken:
            headers["Authorization"] = f"Bearer {hfToken}"

        if not self.ref.repoId:
            return None
        readmeUrl = f"https://huggingface.co/{self.ref.repoId}/raw/main/README.md"
        return _http_get_text(readmeUrl, headers)


class DatasetResource(Resource):
    def fetchMetadata(self) -> Dict[str, Any]:
        headers = {"Accept": "application/json"}
        hfToken = os.environ.get("HUGGINGFACE_TOKEN")
        if hfToken:
            headers["Authorization"] = f"Bearer {hfToken}"

        if not self.ref.repoId:
            LOG.info("Missing repoId for HF dataset")
            return {}

        apiUrl = f"https://huggingface.co/api/datasets/{self.ref.repoId}"
        data = _http_get_json(apiUrl, headers)
        meta = {}
        if data:
            meta = {
                "downloads": data.get("downloads"),
                "likes": data.get("likes"),
                "lastModified": data.get("lastModified"),
                "sha": data.get("sha"),
                "fileCount": len(data.get("siblings") or []),
            }
        return meta

    def fetchReadme(self) -> Optional[str]:
        hfToken = os.environ.get("HUGGINGFACE_TOKEN")
        headers = {"Accept": "text/plain"}
        if hfToken:
            headers["Authorization"] = f"Bearer {hfToken}"

        if not self.ref.repoId:
            return None
        readmeUrl = f"https://huggingface.co/datasets/{self.ref.repoId}/raw/main/README.md"
        return _http_get_text(readmeUrl, headers)


class CodeResource(Resource):
    def fetchMetadata(self) -> Dict[str, Any]:
        headers = {"Accept": "application/vnd.github+json"}
        ghToken = os.environ.get("GITHUB_TOKEN")
        if ghToken:
            headers["Authorization"] = f"Bearer {ghToken}"

        if not self.ref.repoId:
            LOG.info("Missing repoId for GitHub repo")
            return {}

        apiUrl = f"https://api.github.com/repos/{self.ref.repoId}"
        data = _http_get_json(apiUrl, headers)
        meta = {}
        if data:
            licenseInfo = data.get("license") or {}
            meta = {
                "stars": data.get("stargazers_count"),
                "forks": data.get("forks_count"),
                "openIssues": data.get("open_issues_count"),
                "licenseSpdx": licenseInfo.get("spdx_id"),
                "updatedAt": data.get("updated_at"),
                "defaultBranch": data.get("default_branch"),
                "archived": data.get("archived"),
            }
        return meta

    def fetchReadme(self) -> Optional[str]:
        headers = {"Accept": "text/plain"}
        ghToken = os.environ.get("GITHUB_TOKEN")
        if ghToken:
            headers["Authorization"] = f"Bearer {ghToken}"

        if not self.ref.repoId:
            return None
        # Try main first, then master
        for branch in ("main", "master"):
            readmeUrl = f"https://raw.githubusercontent.com/{self.ref.repoId}/{branch}/README.md"
            text = _http_get_text(readmeUrl, headers)
            if text:
                return text
        return None


# ------------------------- URL Classification ------------------------- #

_HF_HOSTS = {"huggingface.co"}
_GH_HOSTS = {"github.com", "www.github.com"}

def _strip_hf(path: str) -> Tuple[UrlCategory, Optional[str], Optional[str]]:
    # /org/name OR /datasets/org/name
    parts = [seg for seg in path.split("/") if seg]
    if not parts:
        return UrlCategory.UNKNOWN, None, None
    if parts[0] == "datasets":
        owner = parts[1] if len(parts) > 1 else None
        name = parts[2] if len(parts) > 2 else None
        return UrlCategory.DATASET, owner, name
    owner = parts[0] if len(parts) > 0 else None
    name = parts[1] if len(parts) > 1 else None
    return UrlCategory.MODEL, owner, name


def _strip_gh(path: str) -> Tuple[Optional[str], Optional[str]]:
    parts = [seg for seg in path.split("/") if seg]
    owner = parts[0] if len(parts) > 0 else None
    name = parts[1] if len(parts) > 1 else None
    return owner, name


def classifyUrl(rawUrl: str) -> ResourceRef:
    """Normalize and classify a URL into a ResourceRef."""
    url = rawUrl.strip()
    try:
        from urllib.parse import urlparse, unquote
        parsed = urlparse(url)
        netloc = (parsed.netloc or "").lower()
        path = unquote(parsed.path or "")
    except Exception:
        netloc = ""
        path = ""

    if netloc in _HF_HOSTS:
        category, owner, name = _strip_hf(path)
        repoId = f"{owner}/{name}" if owner and name else None
        normalizedUrl = (
            f"https://huggingface.co/datasets/{repoId}"
            if category == UrlCategory.DATASET and repoId
            else (f"https://huggingface.co/{repoId}" if repoId else rawUrl)
        )
        host = Host.HUGGINGFACE
        return ResourceRef(url, host, category, owner, name, repoId, normalizedUrl)

    if netloc in _GH_HOSTS:
        owner, name = _strip_gh(path)
        repoId = f"{owner}/{name}" if owner and name else None
        normalizedUrl = f"https://github.com/{repoId}" if repoId else rawUrl
        return ResourceRef(url, Host.GITHUB, UrlCategory.CODE, owner, name, repoId, normalizedUrl)

    return ResourceRef(url, Host.OTHER, UrlCategory.UNKNOWN, None, None, None, rawUrl)


def determineResource(rawUrl: str) -> Resource:
    """Factory that returns the right Resource subclass instance."""
    ref = classifyUrl(rawUrl)
    if ref.host is Host.HUGGINGFACE and ref.category is UrlCategory.MODEL:
        return ModelResource(ref)
    if ref.host is Host.HUGGINGFACE and ref.category is UrlCategory.DATASET:
        return DatasetResource(ref)
    if ref.host is Host.GITHUB and ref.category is UrlCategory.CODE:
        return CodeResource(ref)
    # Fallback: UNKNOWN
    return NoopResource(ref)  # type: ignore[abstract]


# ------------------------- File helper (for ./run URL_FILE path) ------------------------- #

def parseUrlFile(path: str) -> Tuple[ResourceRef, ...]:
    """Read newline-delimited ASCII URLs and classify each line (skip blanks/#)."""
    refs = []
    with open(path, "r", encoding="ascii", errors="strict") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                refs.append(classifyUrl(line))
            except Exception as e:
                LOG.info(f"Skipping URL due to error: {line} ({e})")
    return tuple(refs)


# ------------------------- Simple README helper used by Scorer later ------------------------- #

_LICENSE_HEADING_RE = re.compile(r"^\s*#{1,6}\s*license\b", re.IGNORECASE | re.MULTILINE)

def hasLicenseSection(readmeText: Optional[str]) -> bool:
    if not readmeText:
        return False
    return bool(_LICENSE_HEADING_RE.search(readmeText))


# ------------------------- Public API ------------------------- #

__all__ = [
    "UrlCategory",
    "Host",
    "ResourceRef",
    "Resource",
    "ModelResource",
    "DatasetResource",
    "CodeResource",
    "classifyUrl",
    "determineResource",
    "parseUrlFile",
    "clearCache",
    "hasLicenseSection",
    "NoopResource",
]


# ------------------------- Optional local sanity check ------------------------- #

#if __name__ == "__main__":  # pragma: no cover
#    demoUrls = [
#        "https://huggingface.co/google/gemma-3-270m/tree/main",
#        "https://huggingface.co/datasets/xlangai/AgentNet",
#        "https://github.com/SkyworkAI/Matrix-Game",
#        "https://example.com/something-else",
#    ]
#    for u in demoUrls:
#        r = determineResource(u)
#        meta = {}
#        try:
#            meta = r.fetchMetadata()
#        except TypeError:
            # Abstract fallback for UNKNOWN
#            pass
#        LOG.info(f"demo -> {r.ref.category} {r.ref.repoId} metaKeys={list(meta.keys()) if meta else []}")


# ------------------------- End of File ------------------------- #
'''
Test case :
'''

#resource = determineResource("https://huggingface.co/google/flan-t5-base")
#meta = resource.fetchMetadata()
#readme = resource.fetchReadme()
#print(meta)

