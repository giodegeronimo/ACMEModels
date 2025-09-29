# URL_Fetcher.py
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from enum import Enum, auto
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import HfHubHTTPError

# --- Enums and Data Classes ---
class RefType(Enum):
    MODEL = auto()
    DATASET = auto()
    UNKNOWN = auto()

@dataclass(frozen=True)
class Ref:
    category: RefType
    name: str
    
# --- Regex for URL parsing ---
_RE_HF_MODEL = re.compile(r"https://huggingface.co/([^/]+)/([^/]+)")
_RE_HF_DATASET = re.compile(r"https://huggingface.co/datasets/([^/]+)/([^/]+)")
_RE_LICENSE = re.compile(r"licen[sc]e", re.I)

def determineResource(url: str) -> "Resource":
    """Determines the resource type from a URL and returns a Resource object."""
    url = (url or "").strip()
    if not url:
        return Resource(Ref(RefType.UNKNOWN, url))

    m = _RE_HF_MODEL.match(url)
    if m:
        return Resource(Ref(RefType.MODEL, f"{m.group(1)}/{m.group(2)}"))

    m = _RE_HF_DATASET.match(url)
    if m:
        return Resource(Ref(RefType.DATASET, f"{m.group(1)}/{m.group(2)}"))
    
    return Resource(Ref(RefType.UNKNOWN, url))

def hasLicenseSection(text: str) -> bool:
    """Checks if the text contains a license section."""
    return bool(text and _RE_LICENSE.search(text))

class Resource:
    """Represents a resource (model, dataset, etc.) to be scored."""
    def __init__(self, ref: Ref):
        self.ref = ref
        self._api = HfApi()

    def fetchReadme(self) -> str | None:
        """Fetches the README file for the resource, handling errors gracefully."""
        if self.ref.category not in (RefType.MODEL, RefType.DATASET):
            return None
        try:
            repo_id = self.ref.name
            if self.ref.category == RefType.DATASET:
                repo_id = f"datasets/{repo_id}"
            
            readme_path = hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type=self.ref.category.name.lower(),
                token=os.environ.get("GITHUB_TOKEN"),
            )
            with open(readme_path, "r", encoding="utf-8") as f:
                return f.read()
        except HfHubHTTPError:
            # CRITICAL FIX: This catches authentication errors or file-not-found errors.
            return None
        except Exception:
            # Catch any other unexpected errors during file download or read.
            return None

    def fetchMetadata(self) -> dict | None:
        """Fetches the metadata for the resource, handling errors gracefully."""
        if self.ref.category not in (RefType.MODEL, RefType.DATASET):
            return None
        try:
            repo_id = self.ref.name
            if self.ref.category == RefType.MODEL:
                info = self._api.model_info(repo_id, token=os.environ.get("GITHUB_TOKEN"))
            else: # DATASET
                info = self._api.dataset_info(repo_id, token=os.environ.get("GITHUB_TOKEN"))
            
            # Extract relevant fields into a dictionary
            return {
                "likes": getattr(info, 'likes', 0),
                "downloads": getattr(info, 'downloads', 0),
                "license": getattr(info, 'license', None),
                "fileCount": len(getattr(info, 'siblings', [])),
            }
        except HfHubHTTPError:
            # CRITICAL FIX: This catches authentication errors.
            return None
        except Exception:
            # Catch any other unexpected errors.
            return None
