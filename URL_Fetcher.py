# URL_Fetcher.py
from __future__ import annotations

import re
import json
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional
import requests

_UA = {"User-Agent": "ece461-autograder-compatible/1.0 (+https://purdue.edu)"}

def _safe_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 10.0):
    try:
        h = dict(_UA)
        if headers:
            h.update(headers)
        r = requests.get(url, headers=h, timeout=timeout)
        if r.status_code >= 400:
            return None
        return r
    except Exception:
        return None

def _http_get_json(url: str, headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    r = _safe_get(url, headers=headers)
    if not r:
        return None
    try:
        return r.json()
    except Exception:
        try:
            return json.loads(r.text)
        except Exception:
            return None

class UrlCategory(Enum):
    MODEL = auto()
    DATASET = auto()
    CODE = auto()
    UNKNOWN = auto()

@dataclass(frozen=True)
class Ref:
    name: str
    category: UrlCategory

class Resource:
    def __init__(self, url: str, ref: Ref) -> None:
        self.url = url
        self.ref = ref

    def fetchMetadata(self) -> Dict[str, Any]:
        return {}

    def fetchReadme(self) -> Optional[str]:
        return None

# -------- Hugging Face MODEL --------
_HF_MODEL_RE = re.compile(
    r"^https?://huggingface\.co/(?P<org>[^/]+)/(?P<model>[^/\s#?]+)(?:/tree/[^/]+)?/?$",
    re.IGNORECASE,
)
# Accept bare id (no org) like https://huggingface.co/bert-base-uncased
_HF_MODEL_RE_BARE = re.compile(
    r"^https?://huggingface\.co/(?P<model>[^/\s#?]+)(?:/tree/[^/]+)?/?$",
    re.IGNORECASE,
)

class ModelResource(Resource):
    def __init__(self, url: str, repo_org: Optional[str], repo_name: str):
        ref = Ref(name=repo_name, category=UrlCategory.MODEL)
        super().__init__(url, ref)
        self.org = repo_org
        self.model = repo_name

    @property
    def _repo_id(self) -> str:
        return f"{self.org}/{self.model}" if self.org else self.model

    def fetchMetadata(self) -> Dict[str, Any]:
        api = f"https://huggingface.co/api/models/{self._repo_id}"
        data = _http_get_json(api)
        if not data:
            return {}
        card = data.get("cardData") or {}
        siblings = data.get("siblings") or []
        # SURFACE LICENSE so Scorer.metric_license() can pass
        lic = data.get("license") or card.get("license") or card.get("licenses")
        return {
            "downloads": data.get("downloads"),
            "likes": data.get("likes"),
            "lastModified": data.get("lastModified"),
            "sha": data.get("sha"),
            "fileCount": len(siblings),
            "license": lic,
        }

    def fetchReadme(self) -> Optional[str]:
        # Try common raw paths
        raw_candidates = [
            f"https://huggingface.co/{self._repo_id}/resolve/main/README.md?download=1",
            f"https://huggingface.co/{self._repo_id}/raw/main/README.md",
        ]
        for u in raw_candidates:
            r = _safe_get(u)
            if r and r.text and len(r.text.strip()) > 0 and "DOCTYPE html" not in r.text[:200]:
                return r.text
        # fallback to HTML page (lightly de-tag)
        page = _safe_get(f"https://huggingface.co/{self._repo_id}")
        if page and page.text:
            txt = re.sub(r"<[^>]+>", " ", page.text)
            return re.sub(r"\s+", " ", txt).strip()
        return None

# -------- HF DATASET --------
_HF_DATASET_RE = re.compile(
    r"^https?://huggingface\.co/datasets/(?P<org>[^/]+)/(?P<name>[^/\s#?]+)(?:/tree/[^/]+)?/?$",
    re.IGNORECASE,
)

class DatasetResource(Resource):
    def __init__(self, url: str, repo_org: str, ds_name: str):
        ref = Ref(name=ds_name, category=UrlCategory.DATASET)
        super().__init__(url, ref)
        self.org = repo_org
        self.dataset = ds_name

    def fetchMetadata(self) -> Dict[str, Any]:
        api = f"https://huggingface.co/api/datasets/{self.org}/{self.dataset}"
        data = _http_get_json(api)
        if not data:
            return {}
        siblings = data.get("siblings") or []
        lic = data.get("license") or (data.get("cardData") or {}).get("license")
        return {
            "downloads": data.get("downloads"),
            "likes": data.get("likes"),
            "lastModified": data.get("lastModified"),
            "sha": data.get("sha"),
            "fileCount": len(siblings),
            "license": lic,
        }

    def fetchReadme(self) -> Optional[str]:
        raw_candidates = [
            f"https://huggingface.co/datasets/{self.org}/{self.dataset}/resolve/main/README.md?download=1",
            f"https://huggingface.co/datasets/{self.org}/{self.dataset}/raw/main/README.md",
        ]
        for u in raw_candidates:
            r = _safe_get(u)
            if r and r.text and len(r.text.strip()) > 0 and "DOCTYPE html" not in r.text[:200]:
                return r.text
        page = _safe_get(f"https://huggingface.co/datasets/{self.org}/{self.dataset}")
        if page and page.text:
            txt = re.sub(r"<[^>]+>", " ", page.text)
            return re.sub(r"\s+", " ", txt).strip()
        return None

# -------- GitHub CODE --------
_GH_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/\s#?]+)(?:/.*)?$",
    re.IGNORECASE,
)

class CodeResource(Resource):
    def __init__(self, url: str, owner: str, repo: str):
        ref = Ref(name=repo, category=UrlCategory.CODE)
        super().__init__(url, ref)
        self.owner = owner
        self.repo = repo

    def fetchMetadata(self) -> Dict[str, Any]:
        api = f"https://api.github.com/repos/{self.owner}/{self.repo}"
        data = _http_get_json(api, headers={"Accept": "application/vnd.github+json"})
        if not data:
            return {}
        return {
            "stars": data.get("stargazers_count"),
            "forks": data.get("forks_count"),
            "lastModified": data.get("updated_at"),
            "license": (data.get("license") or {}).get("spdx_id"),
        }

    def fetchReadme(self) -> Optional[str]:
        raw_candidates = [
            f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/main/README.md",
            f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/master/README.md",
        ]
        for u in raw_candidates:
            r = _safe_get(u)
            if r and r.text and len(r.text.strip()) > 0:
                return r.text
        page = _safe_get(f"https://github.com/{self.owner}/{self.repo}")
        if page and page.text:
            txt = re.sub(r"<[^>]+>", " ", page.text)
            return re.sub(r"\s+", " ", txt).strip()
        return None

# -------- Utilities used by Scorer --------
_LIC_HDR_RE = re.compile(r"^\s*#{1,6}\s*license\b", re.IGNORECASE | re.MULTILINE)
_SPDX_RE = re.compile(r"\b(apache-2\.0|mit|bsd-3-clause|gpl-3\.0|mpl-2\.0|lgpl-3\.0|cc-by|cc0)\b", re.IGNORECASE)

def hasLicenseSection(text: Optional[str]) -> bool:
    if not text:
        return False
    if _LIC_HDR_RE.search(text):
        return True
    if _SPDX_RE.search(text):
        return True
    return False

# -------- Router --------
def determineResource(url: str) -> Resource:
    url = (url or "").strip()

    # HF model with org
    m = _HF_MODEL_RE.match(url)
    if m:
        org = m.group("org")
        model = m.group("model")
        return ModelResource(url, org, model)

    # HF model bare id URL
    b = _HF_MODEL_RE_BARE.match(url)
    if b:
        model = b.group("model")
        return ModelResource(url, None, model)

    # HF dataset
    d = _HF_DATASET_RE.match(url)
    if d:
        org = d.group("org")
        name = d.group("name")
        return DatasetResource(url, org, name)

    # GitHub repo
    g = _GH_RE.match(url)
    if g:
        owner = g.group("owner")
        repo = g.group("repo")
        return CodeResource(url, owner, repo)

    # Fallback
    return Resource(url, Ref(name=url, category=UrlCategory.UNKNOWN))
