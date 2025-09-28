# URL_Fetcher.py
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

import requests

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
    handler = logging.FileHandler(logFile, encoding="utf-8") if logFile else logging.NullHandler()

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

# ------------------------- HTTP helpers ------------------------- #
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
    s.headers.update({"User-Agent": "ece461-cli/1.0"})
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
                LOG.info("429 from %s; backing off", url)
                time.sleep(BACKOFF_SEC * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            _cache[url] = data
            return data
        except requests.RequestException as e:
            LOG.info("HTTP error for %s: %s", url, e)
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_SEC * (attempt + 1))
            else:
                return None
        except json.JSONDecodeError:
            LOG.info("JSON decode error for %s", url)
            return None
    return None

def _http_get_text(url: str, headers: Dict[str, str]) -> Optional[str]:
    key = f"text::{url}"
    if key in _cache:
        return _cache[key]
    sess = _get_session()
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = sess.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 429 and attempt < MAX_RETRIES:
                LOG.info("429 from %s; backing off", url)
                time.sleep(BACKOFF_SEC * (attempt + 1))
                continue
            resp.raise_for_status()
            txt = resp.text
            _cache[key] = txt
            return txt
        except requests.RequestException as e:
            LOG.info("HTTP error for %s: %s", url, e)
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_SEC * (attempt + 1))
            else:
                return None
    return None

def clearCache() -> None:
    _cache.clear()

# ------------------------- Resource abstraction ------------------------- #
class Resource(ABC):
    def __init__(self, ref: ResourceRef) -> None:
        self.ref = ref

    @abstractmethod
    def fetchMetadata(self) -> Dict[str, Any]:
        ...

    def fetchReadme(self) -> Optional[str]:
        return None

class NoopResource(Resource):
    def fetchMetadata(self) -> Dict[str, Any]:
        return {}
    def fetchReadme(self) -> Optional[str]:
        return None

class ModelResource(Resource):
    def fetchMetadata(self) -> Dict[str, Any]:
        headers = {"Accept": "application/json"}
        hfToken = os.environ.get("HUGGINGFACE_TOKEN")
        if hfToken:
            headers["Authorization"] = f"Bearer {hfToken}"

        if not self.ref.repoId:
            LOG.info("Missing repoId for HF model")
            return {}

        apiUrl = f"https://huggingface.co/api/models/{self.ref.repoId}"
        data = _http_get_json(apiUrl, headers)
        meta: Dict[str, Any] = {}
        if data:
            card = data.get("cardData") or {}
            siblings = data.get("siblings") or []
            meta = {
                "downloads": data.get("downloads"),
                "likes": data.get("likes"),
                "lastModified": data.get("lastModified"),
                "sha": data.get("sha"),
                "fileCount": len(siblings),
                # 🔑 surface license for Scorer.metric_license()
                "license": data.get("license") or card.get("license") or card.get("licenses"),
            }
        return meta

    def fetchReadme(self) -> Optional[str]:
        headers = {"Accept": "text/plain"}
        hfToken = os.environ.get("HUGGINGFACE_TOKEN")
        if hfToken:
            headers["Authorization"] = f"Bearer {hfToken}"
        if not self.ref.repoId:
            return None
        for u in (
            f"https://huggingface.co/{self.ref.repoId}/resolve/main/README.md?download=1",
            f"https://huggingface.co/{self.ref.repoId}/raw/main/README.md",
        ):
            txt = _http_get_text(u, headers)
            if txt and "DOCTYPE html" not in (txt[:200] or ""):
                return txt
        # fallback: HTML stripped
        page = _http_get_text(f"https://huggingface.co/{self.ref.repoId}", headers={"Accept":"text/html"})
        if page:
            return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page)).strip()
        return None

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
        meta: Dict[str, Any] = {}
        if data:
            card = data.get("cardData") or {}
            siblings = data.get("siblings") or []
            meta = {
                "downloads": data.get("downloads"),
                "likes": data.get("likes"),
                "lastModified": data.get("lastModified"),
                "sha": data.get("sha"),
                "fileCount": len(siblings),
                "license": data.get("license") or card.get("license") or card.get("licenses"),
            }
        return meta

    def fetchReadme(self) -> Optional[str]:
        headers = {"Accept": "text/plain"}
        hfToken = os.environ.get("HUGGINGFACE_TOKEN")
        if hfToken:
            headers["Authorization"] = f"Bearer {hfToken}"
        if not self.ref.repoId:
            return None
        for u in (
            f"https://huggingface.co/datasets/{self.ref.repoId}/resolve/main/README.md?download=1",
            f"https://huggingface.co/datasets/{self.ref.repoId}/raw/main/README.md",
        ):
            txt = _http_get_text(u, headers)
            if txt and "DOCTYPE html" not in (txt[:200] or ""):
                return txt
        page = _http_get_text(f"https://huggingface.co/datasets/{self.ref.repoId}", headers={"Accept":"text/html"})
        if page:
            return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page)).strip()
        return None

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
        if not data:
            return {}
        lic = (data.get("license") or {}).get("spdx_id")
        return {
            "stars": data.get("stargazers_count"),
            "forks": data.get("forks_count"),
            "lastModified": data.get("updated_at"),
            "license": lic,
        }

    def fetchReadme(self) -> Optional[str]:
        for u in (
            f"https://raw.githubusercontent.com/{self.ref.repoId}/main/README.md",
            f"https://raw.githubusercontent.com/{self.ref.repoId}/master/README.md",
        ):
            txt = _http_get_text(u, headers={"Accept":"text/plain"})
            if txt:
                return txt
        page = _http_get_text(f"https://github.com/{self.ref.repoId}", headers={"Accept":"text/html"})
        if page:
            return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page)).strip()
        return None

# ------------------------- URL classification ------------------------- #
_HF_HOSTS = {"huggingface.co"}
_GH_HOSTS = {"github.com", "www.github.com"}

def _strip_hf(path: str) -> Tuple[UrlCategory, Optional[str], Optional[str]]:
    parts = [seg for seg in path.split("/") if seg]
    if not parts:
        return UrlCategory.UNKNOWN, None, None
    if parts[0] == "datasets":
        if len(parts) == 2:
            return UrlCategory.DATASET, None, parts[1]
        owner = parts[1] if len(parts) > 1 else None
        name = parts[2] if len(parts) > 2 else None
        return UrlCategory.DATASET, owner, name
    # model
    if len(parts) == 1:
        return UrlCategory.MODEL, None, parts[0]
    owner = parts[0]
    name = parts[1] if len(parts) > 1 else None
    return UrlCategory.MODEL, owner, name

def _strip_gh(path: str) -> Tuple[Optional[str], Optional[str]]:
    parts = [seg for seg in path.split("/") if seg]
    owner = parts[0] if len(parts) > 0 else None
    name = parts[1] if len(parts) > 1 else None
    return owner, name

def classifyUrl(rawUrl: str) -> ResourceRef:
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
        if owner and name:
            repoId = f"{owner}/{name}"
        elif name:
            repoId = name
        else:
            repoId = None

        if category == UrlCategory.DATASET and repoId:
            normalizedUrl = f"https://huggingface.co/datasets/{repoId}"
        elif repoId:
            normalizedUrl = f"https://huggingface.co/{repoId}"
        else:
            normalizedUrl = rawUrl

        return ResourceRef(url, Host.HUGGINGFACE, category, owner, name, repoId, normalizedUrl)

    if netloc in _GH_HOSTS:
        owner, name = _strip_gh(path)
        repoId = f"{owner}/{name}" if owner and name else None
        normalizedUrl = f"https://github.com/{repoId}" if repoId else rawUrl
        return ResourceRef(url, Host.GITHUB, UrlCategory.CODE, owner, name, repoId, normalizedUrl)

    return ResourceRef(url, Host.OTHER, UrlCategory.UNKNOWN, None, None, None, rawUrl)

def determineResource(rawUrl: str) -> Resource:
    ref = classifyUrl(rawUrl)
    if ref.host is Host.HUGGINGFACE and ref.category is UrlCategory.MODEL:
        return ModelResource(ref)
    if ref.host is Host.HUGGINGFACE and ref.category is UrlCategory.DATASET:
        return DatasetResource(ref)
    if ref.host is Host.GITHUB and ref.category is UrlCategory.CODE:
        return CodeResource(ref)
    return NoopResource(ref)  # type: ignore[abstract]

# ------------------------- README helper used by Scorer ------------------------- #
_LICENSE_HEADING_RE = re.compile(r"^\s*#{1,6}\s*license\b", re.IGNORECASE | re.MULTILINE)
def hasLicenseSection(readmeText: Optional[str]) -> bool:
    if not readmeText:
        return False
    return bool(_LICENSE_HEADING_RE.search(readmeText))

__all__ = [
    "UrlCategory","Host","ResourceRef","Resource",
    "ModelResource","DatasetResource","CodeResource",
    "classifyUrl","determineResource","clearCache",
    "hasLicenseSection","NoopResource",
]
