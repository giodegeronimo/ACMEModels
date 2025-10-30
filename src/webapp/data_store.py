"""Lightweight in-memory datastore backing the prototype web UI."""

from __future__ import annotations

import copy
import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class ModelRecord:
    """Representation of a model in the registry prototype."""

    model_id: str
    name: str
    description: str
    card_excerpt: str
    owner: str
    vetted: bool
    tags: List[str]
    license_id: str
    updated_at: float
    size_mb: float
    download_url: str
    card_url: str
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


_DEFAULT_MODELS: Tuple[ModelRecord, ...] = (
    ModelRecord(
        model_id="acme/solar-safeguard",
        name="Solar Safeguard",
        description="Detects and prioritizes solar array maintenance tasks.",
        card_excerpt=(
            "Optimized vision transformer designed for edge deployments on "
            "industrial solar installations."
        ),
        owner="ACME Internal",
        vetted=True,
        tags=["vision", "maintenance", "edge"],
        license_id="acme-proprietary",
        updated_at=time.time() - 86_400,
        size_mb=512.0,
        download_url="https://registry.acme.example/solar-safeguard",
        card_url="https://registry.acme.example/cards/solar-safeguard",
        parents=["hf/google/vit-small-patch16-224"],
        children=["acme/solar-safeguard-edge"],
        metrics={
            "net_score": 0.82,
            "ramp_up_time": 0.91,
            "bus_factor": 0.7,
            "performance_claims": 0.77,
            "license": 1.0,
            "size_score": 0.68,
            "dataset_and_code_score": 0.72,
        },
    ),
    ModelRecord(
        model_id="acme/solar-safeguard-edge",
        name="Solar Safeguard Edge",
        description=(
            "Quantized derivative of Solar", ""
            "Safeguard for Jetson-class devices.",
        ),
        card_excerpt=(
            "Fine-tuned on edge telemetry to run efficiently on Jetson Nano "
            "and Raspberry Pi compute modules."
        ),
        owner="ACME Internal",
        vetted=False,
        tags=["vision", "maintenance", "edge", "quantized"],
        license_id="acme-proprietary",
        updated_at=time.time() - 21_600,
        size_mb=164.0,
        download_url="https://registry.acme.example/solar-safeguard-edge",
        card_url="https://registry.acme.example/cards/solar-safeguard-edge",
        parents=["acme/solar-safeguard"],
        metrics={
            "net_score": 0.79,
            "ramp_up_time": 0.88,
            "bus_factor": 0.65,
            "performance_claims": 0.71,
            "license": 1.0,
            "size_score": 0.89,
            "dataset_and_code_score": 0.67,
        },
    ),
    ModelRecord(
        model_id="hf/google/vit-small-patch16-224",
        name="ViT Small Patch16 224",
        description="Public vision transformer base model from Google.",
        card_excerpt=(
            "General vision transformer "
            "architecture pre-trained on ImageNet21k."
        ),
        owner="Google",
        vetted=True,
        tags=["vision", "transformer"],
        license_id="apache-2.0",
        updated_at=time.time() - (5 * 86_400),
        size_mb=336.0,
        download_url="https://huggingface.co/google/vit-small-patch16-224",
        card_url="https://huggingface.co/google/vit-small-patch16-224#card",
        children=["acme/solar-safeguard"],
        metrics={
            "net_score": 0.75,
            "ramp_up_time": 0.83,
            "bus_factor": 0.92,
            "performance_claims": 0.66,
            "license": 1.0,
            "size_score": 0.54,
            "dataset_and_code_score": 0.62,
        },
    ),
    ModelRecord(
        model_id="hf/open-catalyst/oc20-gnn",
        name="OC20 Catalyst GNN",
        description="Predicts adsorption energies for surface reactions.",
        card_excerpt="Graph neural network trained on the OC20 dataset.",
        owner="Open Catalyst Project",
        vetted=False,
        tags=["chemistry", "gnn"],
        license_id="mit",
        updated_at=time.time() - (8 * 86_400),
        size_mb=2_048.0,
        download_url="https://huggingface.co/open-catalyst/oc20-gnn",
        card_url="https://huggingface.co/open-catalyst/oc20-gnn#card",
        metrics={
            "net_score": 0.64,
            "ramp_up_time": 0.59,
            "bus_factor": 0.48,
            "performance_claims": 0.62,
            "license": 0.92,
            "size_score": 0.41,
            "dataset_and_code_score": 0.58,
        },
    ),
)


def _load_license_catalog() -> Dict[str, set[str]]:
    """Load license compatibility data shipped with the repo."""

    catalog = {
        "compatible": set(),
        "caution": set(),
        "incompatible": set(),
    }

    mapping = {
        "compatible": "licenses_compatible_spdx.json",
        "caution": "licenses_caution_spdx.json",
        "incompatible": "licenses_incompatible_spdx.json",
    }

    for key, filename in mapping.items():
        path = DATA_DIR / filename
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            continue

        items = payload if isinstance(payload, list) else payload.get("items", [])
        catalog[key] = {item.lower() for item in items if isinstance(item, str)}

    return catalog


class DataStore:
    """Prototype state holder with methods that mimic future integrations."""

    def __init__(self) -> None:
        self._license_catalog = _load_license_catalog()
        self._default_models = {model.model_id: model for model in _DEFAULT_MODELS}
        self._models: Dict[str, ModelRecord] = {}
        self._ingest_requests: List[Dict[str, object]] = []
        self.reset()

    # ------------------------------------------------------------------
    # Model directory
    # ------------------------------------------------------------------
    def list_models(
        self,
        *,
        limit: int = 50,
        cursor: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> Tuple[List[Dict[str, object]], Optional[str], int]:
        """Return paginated models filtered by optional regex pattern."""
        limit = max(1, min(limit, 100))
        offset = 0
        if cursor:
            try:
                offset = int(cursor)
            except ValueError:
                offset = 0

        models = list(self._models.values())
        if pattern:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error:
                raise ValueError("Invalid regular expression supplied.")
            models = [
                record
                for record in models
                if regex.search(record.name)
                or regex.search(record.card_excerpt)
                or regex.search(record.model_id)
            ]

        total = len(models)
        models.sort(key=lambda record: record.name.lower())

        slice_end = min(offset + limit, total)
        page = [
            {
                "id": record.model_id,
                "name": record.name,
                "owner": record.owner,
                "vetted": record.vetted,
                "tags": record.tags,
                "updated_at": record.updated_at,
                "license": record.license_id,
                "size_mb": record.size_mb,
            }
            for record in models[offset:slice_end]
        ]

        next_cursor = str(slice_end) if slice_end < total else None
        return page, next_cursor, total

    def get_model(self, model_id: str) -> Optional[Dict[str, object]]:
        """Return a single model, or None if not found."""
        record = self._models.get(model_id)
        if record is None:
            return None

        return {
            "id": record.model_id,
            "name": record.name,
            "description": record.description,
            "card_excerpt": record.card_excerpt,
            "owner": record.owner,
            "vetted": record.vetted,
            "tags": record.tags,
            "license": record.license_id,
            "updated_at": record.updated_at,
            "size_mb": record.size_mb,
            "download_url": record.download_url,
            "card_url": record.card_url,
            "parents": record.parents,
            "children": record.children,
            "metrics": record.metrics,
        }

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_model(
        self,
        *,
        name: str,
        source_url: str,
        metrics: Dict[str, float],
        submitted_by: str,
    ) -> Dict[str, object]:
        """Register an ingestion request if metrics meet minimum thresholds."""
        required_metrics = (
            "net_score",
            "ramp_up_time",
            "bus_factor",
            "performance_claims",
            "license",
            "size_score",
            "dataset_and_code_score",
        )

        missing = [metric for metric in required_metrics if metric not in metrics]
        if missing:
            return {
                "accepted": False,
                "reason": f"Missing metrics: {', '.join(sorted(missing))}",
            }

        insufficient = {
            key: value for key, value in metrics.items() if key in required_metrics and value < 0.5
        }
        if insufficient:
            return {
                "accepted": False,
                "reason": (
                    "Scores below 0.5 detected: "
                    + ", ".join(f"{metric}={score:.2f}" for metric, score in insufficient.items())
                ),
            }

        request_id = str(uuid.uuid4())
        payload = {
            "request_id": request_id,
            "name": name,
            "source_url": source_url,
            "metrics": metrics,
            "submitted_by": submitted_by,
            "status": "queued",
            "submitted_at": time.time(),
        }
        self._ingest_requests.append(payload)
        return {"accepted": True, "request_id": request_id}

    # ------------------------------------------------------------------
    # Lineage and size
    # ------------------------------------------------------------------
    def lineage(self, model_id: str) -> Optional[Dict[str, object]]:
        """Produce a lineage graph representation for a model."""
        model = self._models.get(model_id)
        if model is None:
            return None

        def _node_payload(record: ModelRecord) -> Dict[str, object]:
            return {
                "id": record.model_id,
                "name": record.name,
                "vetted": record.vetted,
                "license": record.license_id,
            }

        nodes = {_node_payload(model)["id"]: _node_payload(model)}

        edges: List[Dict[str, str]] = []
        for parent_id in model.parents:
            parent = self._models.get(parent_id)
            if parent:
                nodes[parent.model_id] = _node_payload(parent)
                edges.append({"from": parent.model_id, "to": model.model_id})
        for child_id in model.children:
            child = self._models.get(child_id)
            if child:
                nodes[child.model_id] = _node_payload(child)
                edges.append({"from": model.model_id, "to": child.model_id})

        return {
            "root": model.model_id,
            "nodes": list(nodes.values()),
            "edges": edges,
        }

    def size_cost(self, model_id: str) -> Optional[Dict[str, object]]:
        """Return a rough cost heuristic based on download size."""
        model = self._models.get(model_id)
        if model is None:
            return None

        size_gb = model.size_mb / 1024
        download_minutes = math.ceil(size_gb * 2.5)
        tiers = {
            "edge_device": min(1.0, max(0.0, 1 - (model.size_mb / 512))),
            "workstation": min(1.0, max(0.0, 1 - (model.size_mb / 2048))),
            "cloud_gpu": min(1.0, max(0.0, 1 - (model.size_mb / 8192))),
        }

        return {
            "model_id": model.model_id,
            "size_mb": model.size_mb,
            "estimated_download_minutes": download_minutes,
            "capacity_score": tiers,
        }

    # ------------------------------------------------------------------
    # License compatibility
    # ------------------------------------------------------------------
    def assess_license(self, repo_license: str, model_license: str) -> Dict[str, object]:
        """Assess whether two licenses are compatible for fine-tune + inference."""
        repo = repo_license.lower()
        model = model_license.lower()

        compatible = self._license_catalog["compatible"]
        caution = self._license_catalog["caution"]
        incompatible = self._license_catalog["incompatible"]

        def _bucket(license_id: str) -> str:
            if license_id in compatible:
                return "compatible"
            if license_id in incompatible:
                return "incompatible"
            if license_id in caution:
                return "caution"
            return "unknown"

        repo_bucket = _bucket(repo)
        model_bucket = _bucket(model)

        if "incompatible" in (repo_bucket, model_bucket):
            status = "incompatible"
            rationale = (
                "One or more licenses are explicitly incompatible with commercial reuse."
            )
        elif repo_bucket == "unknown" or model_bucket == "unknown":
            status = "needs_review"
            rationale = "Automatic assessment unavailable; manual review required."
        elif repo_bucket == "caution" or model_bucket == "caution":
            status = "needs_review"
            rationale = (
                "At least one license carries additional obligations. "
                "Legal review recommended."
            )
        else:
            status = "compatible"
            rationale = "Licenses appear compatible for fine-tune and inference."

        return {
            "compatibility": status,
            "repo_license_bucket": repo_bucket,
            "model_license_bucket": model_bucket,
            "rationale": rationale,
        }

    # ------------------------------------------------------------------
    # Admin / maintenance
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset to default models and clear transient state."""
        self._models = {key: copy.deepcopy(value) for key, value in self._default_models.items()}
        self._ingest_requests.clear()

    def stats(self) -> Dict[str, object]:
        """Return summary statistics for dashboards."""
        total = len(self._models)
        vetted = sum(1 for model in self._models.values() if model.vetted)
        unvetted = total - vetted
        proprietary = sum(
            1 for model in self._models.values() if model.license_id.startswith("acme")
        )
        public = total - proprietary
        return {
            "total_models": total,
            "vetted": vetted,
            "unvetted": unvetted,
            "proprietary": proprietary,
            "public": public,
        }

    def recent_ingest_requests(self) -> List[Dict[str, object]]:
        """Return a copy of recent ingest requests."""
        return list(self._ingest_requests[-10:])
