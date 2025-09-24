# LLM_Analyzer.py
"""
Optional LLM-based analysis for README + metadata (Phase-1 requirement).
If configured, we will call an external endpoint (e.g., Amazon SageMaker endpoint,
Purdue GenAI Studio pipeline, or your own FastAPI) to extract structured facts
used by metrics.

Configuration (environment variables):
- GENAI_ENDPOINT: HTTPS endpoint that accepts POST application/json with:
    { "readme": "<string>", "metadata": { ... }, "instruction": "<prompt>" }
  and returns JSON like:
    {
      "has_examples": true,
      "has_benchmarks": "third_party" | "self_reported" | "vague" | "none",
      "license_name": "MIT",
      "has_dataset_links": true,
      "has_code_links": true
    }
- GENAI_API_KEY: (optional) bearer token for Authorization header

If no endpoint is configured or the call fails, we fall back to light local heuristics.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

try:
    import requests
except Exception:
    requests = None  # type: ignore

_BENCH_RE = re.compile(
    r"(benchmark|results?|accuracy|f1|bleu|rouge|mmlu|leaderboard|eval|evaluation)",
    re.IGNORECASE,
)
_DATASET_LINK_RE = re.compile(r"https?://huggingface\.co/(datasets/|.*\bdata)", re.IGNORECASE)
_CODE_LINK_RE = re.compile(r"https?://(github\.com|gitlab\.com)/", re.IGNORECASE)
_EXAMPLE_RE = re.compile(r"```|pip install|from\s+\w+\s+import|Usage:|#\s*Example", re.IGNORECASE)
_LICENSE_NAME_RE = re.compile(r"license[:\s]*([A-Za-z0-9\-\._ ]+)", re.IGNORECASE)

def _fallback_rules(readme: Optional[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    text = readme or ""
    has_examples = bool(_EXAMPLE_RE.search(text))
    # classify benchmark strength
    if not text or not _BENCH_RE.search(text):
        bench = "none"
    else:
        bench = "vague"
        # crude: presence of a markdown table suggests metrics, bump to self_reported
        if "|" in text and "---" in text:
            bench = "self_reported"
        # links to external leaderboards/papers (very rough third_party bump)
        if re.search(r"paperswithcode|arxiv\.org|leaderboard", text, re.IGNORECASE):
            bench = "third_party"
    m = _LICENSE_NAME_RE.search(text)
    license_name = (m.group(1).strip() if m else None) or ""
    return {
        "has_examples": has_examples,
        "has_benchmarks": bench,
        "license_name": license_name,
        "has_dataset_links": bool(_DATASET_LINK_RE.search(text)),
        "has_code_links": bool(_CODE_LINK_RE.search(text)),
    }

def analyze_readme_and_metadata(readme: Optional[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    ep = os.environ.get("GENAI_ENDPOINT", "").strip()
    if not ep or requests is None:
        return _fallback_rules(readme, metadata)

    payload = {
        "readme": readme or "",
        "metadata": metadata or {},
        "instruction": (
            "Extract facts from the README to help score: "
            "1) has_examples (bool), "
            "2) has_benchmarks (third_party|self_reported|vague|none), "
            "3) license_name (short SPDX-like string if present), "
            "4) has_dataset_links (bool), "
            "5) has_code_links (bool). "
            "Return *only* this JSON object."
        ),
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("GENAI_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        r = requests.post(ep, headers=headers, data=json.dumps(payload), timeout=15)
        r.raise_for_status()
        data = r.json()
        # Validate / coerce expected keys
        out = _fallback_rules(readme, metadata)
        if isinstance(data, dict):
            out["has_examples"] = bool(data.get("has_examples", out["has_examples"]))
            hb = str(data.get("has_benchmarks", out["has_benchmarks"])).lower()
            out["has_benchmarks"] = hb if hb in {"third_party","self_reported","vague","none"} else out["has_benchmarks"]
            lic = str(data.get("license_name", out["license_name"])).strip()
            out["license_name"] = lic
            out["has_dataset_links"] = bool(data.get("has_dataset_links", out["has_dataset_links"]))
            out["has_code_links"] = bool(data.get("has_code_links", out["has_code_links"]))
        return out
    except Exception:
        # Never crash scoring if endpoint fails
        return _fallback_rules(readme, metadata)
