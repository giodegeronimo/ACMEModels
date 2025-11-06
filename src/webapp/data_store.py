"""Lightweight in-memory datastore backing the prototype web UI."""

from __future__ import annotations

import copy
import json
import math
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SUPPORTED_ARTIFACT_TYPES = {"model", "dataset", "code"}
IMPLEMENTED_ARTIFACT_TYPES = {"model"}


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

    def to_metadata(self) -> Dict[str, Any]:
        """Return artifact metadata compatible with the OpenAPI spec."""
        return {
            "name": self.name,
            "id": self.model_id,
            "type": "model",
            "owner": self.owner,
            "vetted": self.vetted,
            "license": self.license_id,
            "updated_at": self.updated_at,
            "size_mb": self.size_mb,
            "tags": list(self.tags),
        }

    def to_model_payload(self) -> Dict[str, Any]:
        """Return the legacy model payload used by the UI templates."""
        return {
            "id": self.model_id,
            "name": self.name,
            "description": self.description,
            "card_excerpt": self.card_excerpt,
            "owner": self.owner,
            "vetted": self.vetted,
            "tags": list(self.tags),
            "license": self.license_id,
            "updated_at": self.updated_at,
            "size_mb": self.size_mb,
            "download_url": self.download_url,
            "card_url": self.card_url,
            "parents": list(self.parents),
            "children": list(self.children),
            "metrics": dict(self.metrics),
        }

    def to_artifact_envelope(self) -> Dict[str, Any]:
        """Return an artifact envelope aligning with the OpenAPI schema."""
        metadata = self.to_metadata()
        metadata.pop("updated_at", None)
        metadata.pop("size_mb", None)
        metadata.pop("tags", None)
        return {
            "metadata": {
                **metadata,
                "updated_at": self.updated_at,
                "size_mb": self.size_mb,
                "tags": list(self.tags),
            },
            "data": {
                "url": self.card_url,
                "download_url": self.download_url,
                "description": self.description,
                "card_excerpt": self.card_excerpt,
                "owner": self.owner,
                "vetted": self.vetted,
                "metrics": dict(self.metrics),
                "parents": list(self.parents),
                "children": list(self.children),
            },
        }


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

        return record.to_model_payload()

    # ------------------------------------------------------------------
    # Artifact API helpers (Phase 2 spec alignment)
    # ------------------------------------------------------------------
    def _normalize_type(self, artifact_type: str) -> str:
        normalized = str(artifact_type or "").strip().lower()
        if not normalized:
            raise ValueError("artifact_type is required.")
        if normalized not in SUPPORTED_ARTIFACT_TYPES:
            raise ValueError(f"Unsupported artifact type '{artifact_type}'.")
        return normalized

    def _require_model_type(self, artifact_type: str) -> str:
        normalized = self._normalize_type(artifact_type)
        if normalized not in IMPLEMENTED_ARTIFACT_TYPES:
            raise NotImplementedError(
                f"Artifact type '{artifact_type}' is not implemented in this prototype."
            )
        return normalized

    def _iter_models(self) -> Iterable[ModelRecord]:
        return self._models.values()

    def _wildcard_to_regex(self, pattern: str) -> re.Pattern:
        escaped = re.escape(pattern)
        escaped = escaped.replace(r"\*", ".*").replace(r"\?", ".")
        return re.compile(f"^{escaped}$", re.IGNORECASE)

    def list_artifacts(
        self,
        queries: Sequence[Mapping[str, Any]],
        *,
        offset: int = 0,
        limit: int = 50,
    ) -> Tuple[List[Dict[str, Any]], Optional[int], int]:
        if not queries:
            raise ValueError("At least one artifact query must be provided.")

        try:
            limit_val = int(limit)
        except (TypeError, ValueError):
            limit_val = 50
        limit_val = max(1, min(limit_val, 100))

        try:
            offset_val = int(offset)
        except (TypeError, ValueError):
            offset_val = 0
        offset_val = max(0, offset_val)

        matched: Dict[str, ModelRecord] = {}
        models = list(self._iter_models())

        for query in queries:
            name_raw = str(query.get("name", "")).strip()
            if not name_raw:
                raise ValueError("Each artifact query must include a non-empty name.")

            type_filters = query.get("types") or []
            normalized_types = {
                self._normalize_type(artifact_type)
                for artifact_type in type_filters
                if artifact_type is not None
            }
            if normalized_types and IMPLEMENTED_ARTIFACT_TYPES.isdisjoint(normalized_types):
                continue

            candidates = models if (not normalized_types or "model" in normalized_types) else []

            if name_raw == "*":
                for record in candidates:
                    matched[record.model_id] = record
                continue

            matcher: re.Pattern | None = None
            if any(ch in name_raw for ch in "*?"):
                try:
                    matcher = self._wildcard_to_regex(name_raw)
                except re.error as exc:
                    raise ValueError("Invalid wildcard expression supplied.") from exc

            for record in candidates:
                if matcher and matcher.search(record.name):
                    matched[record.model_id] = record
                elif not matcher and record.name.lower() == name_raw.lower():
                    matched[record.model_id] = record

        ordered = sorted(
            matched.values(),
            key=lambda record: (record.name.lower(), record.model_id),
        )
        total = len(ordered)
        slice_end = min(offset_val + limit_val, total)
        page = [record.to_metadata() for record in ordered[offset_val:slice_end]]
        next_offset = slice_end if slice_end < total else None
        return page, next_offset, total

    def get_artifact(self, artifact_type: str, artifact_id: str) -> Optional[Dict[str, Any]]:
        self._require_model_type(artifact_type)
        record = self._models.get(artifact_id)
        if record is None:
            return None
        return record.to_artifact_envelope()

    def create_artifact(self, artifact_type: str, data: Mapping[str, Any]) -> Dict[str, Any]:
        self._require_model_type(artifact_type)
        if not isinstance(data, Mapping):
            raise ValueError("Artifact data must be an object.")

        url = str(data.get("url", "")).strip()
        if not url:
            raise ValueError("url is required.")
        download_url = str(data.get("download_url") or url).strip()

        name = str(data.get("name") or "").strip()
        parsed = urlparse(url)
        slug = Path(parsed.path.rstrip("/") or "/").name
        if not slug:
            slug = parsed.path.strip("/").split("/")[-1] if parsed.path else ""
        if not name:
            name = slug or f"{artifact_type}-{uuid.uuid4().hex[:8]}"

        artifact_id = str(data.get("id") or uuid.uuid4().hex)
        if artifact_id in self._models:
            raise FileExistsError("Artifact exists already.")

        description = str(data.get("description") or f"Ingested artifact for {name}.")
        card_excerpt = str(data.get("card_excerpt") or "Ingested via ArtifactCreate API.")
        owner = str(data.get("owner") or "external")
        license_id = str(data.get("license") or "unknown")
        tags = list(data.get("tags") or [])
        parents = list(data.get("parents") or [])
        children = list(data.get("children") or [])
        metrics = dict(data.get("metrics") or {})
        try:
            size_mb = float(data.get("size_mb") or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("size_mb must be numeric.") from exc

        record = ModelRecord(
            model_id=artifact_id,
            name=name,
            description=description,
            card_excerpt=card_excerpt,
            owner=owner,
            vetted=False,
            tags=tags,
            license_id=license_id,
            updated_at=time.time(),
            size_mb=size_mb,
            download_url=download_url,
            card_url=url,
            parents=parents,
            children=children,
            metrics=metrics,
        )
        self._models[artifact_id] = record
        return record.to_artifact_envelope()

    def update_artifact(
        self,
        artifact_type: str,
        artifact_id: str,
        artifact: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        self._require_model_type(artifact_type)
        record = self._models.get(artifact_id)
        if record is None:
            return None
        if not isinstance(artifact, Mapping):
            raise ValueError("Artifact payload must be an object.")

        metadata = artifact.get("metadata") or {}
        data = artifact.get("data") or {}
        if not isinstance(metadata, Mapping) or not isinstance(data, Mapping):
            raise ValueError("Artifact payload must include metadata and data objects.")

        incoming_id = metadata.get("id")
        if incoming_id and incoming_id != artifact_id:
            raise ValueError("metadata.id must match the artifact id.")

        if "name" in metadata:
            record.name = str(metadata["name"])
        if "owner" in metadata:
            record.owner = str(metadata["owner"])
        if "vetted" in metadata:
            record.vetted = bool(metadata["vetted"])
        if "license" in metadata:
            record.license_id = str(metadata["license"])
        if "tags" in metadata:
            record.tags = list(metadata["tags"] or [])
        if "size_mb" in metadata:
            try:
                record.size_mb = float(metadata["size_mb"])
            except (TypeError, ValueError) as exc:
                raise ValueError("size_mb must be numeric.") from exc

        if "url" in data:
            record.card_url = str(data["url"])
        if "download_url" in data:
            record.download_url = str(data["download_url"])
        if "description" in data:
            record.description = str(data["description"])
        if "card_excerpt" in data:
            record.card_excerpt = str(data["card_excerpt"])
        if "metrics" in data:
            record.metrics = dict(data["metrics"] or {})
        if "parents" in data:
            record.parents = list(data["parents"] or [])
        if "children" in data:
            record.children = list(data["children"] or [])

        record.updated_at = time.time()
        self._models[artifact_id] = record
        return record.to_artifact_envelope()

    def delete_artifact(self, artifact_type: str, artifact_id: str) -> bool:
        self._require_model_type(artifact_type)
        return self._models.pop(artifact_id, None) is not None

    def artifact_rating(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        record = self._models.get(artifact_id)
        if record is None:
            return None

        metrics = record.metrics or {}

        def metric_value(key: str, default: float) -> float:
            try:
                return float(metrics.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        size_score_value = metric_value("size_score", 0.72)
        dataset_score = metric_value("dataset_and_code_score", 0.7)
        latency = 0.05

        size_score = {
            "raspberry_pi": max(0.0, min(1.0, size_score_value)),
            "jetson_nano": max(0.0, min(1.0, size_score_value * 0.95 + 0.03)),
            "desktop_pc": max(0.0, min(1.0, size_score_value * 0.9 + 0.05)),
            "aws_server": max(0.0, min(1.0, size_score_value * 0.85 + 0.07)),
        }

        return {
            "name": record.name,
            "category": next(iter(record.tags), "model"),
            "net_score": metric_value("net_score", 0.75),
            "net_score_latency": latency,
            "ramp_up_time": metric_value("ramp_up_time", 0.8),
            "ramp_up_time_latency": latency,
            "bus_factor": metric_value("bus_factor", 0.7),
            "bus_factor_latency": latency,
            "performance_claims": metric_value("performance_claims", 0.7),
            "performance_claims_latency": latency,
            "license": metric_value("license", 0.8),
            "license_latency": latency,
            "dataset_and_code_score": dataset_score,
            "dataset_and_code_score_latency": latency,
            "dataset_quality": dataset_score,
            "dataset_quality_latency": latency,
            "code_quality": dataset_score,
            "code_quality_latency": latency,
            "reproducibility": metric_value("net_score", 0.75),
            "reproducibility_latency": latency,
            "reviewedness": metric_value("performance_claims", 0.7),
            "reviewedness_latency": latency,
            "tree_score": metric_value("bus_factor", 0.7),
            "tree_score_latency": latency,
            "size_score": size_score,
            "size_score_latency": latency,
        }

    def artifact_cost(
        self,
        artifact_type: str,
        artifact_id: str,
        include_dependencies: bool,
    ) -> Optional[Dict[str, Any]]:
        self._require_model_type(artifact_type)
        record = self._models.get(artifact_id)
        if record is None:
            return None

        result: Dict[str, Dict[str, float]] = {
            artifact_id: {"total_cost": float(record.size_mb)}
        }

        if include_dependencies:
            related_ids = set(record.parents + record.children)
            total_cost = float(record.size_mb)
            for related_id in related_ids:
                related = self._models.get(related_id)
                if not related:
                    continue
                result[related.model_id] = {
                    "standalone_cost": float(related.size_mb),
                    "total_cost": float(related.size_mb),
                }
                total_cost += float(related.size_mb)
            result[artifact_id]["standalone_cost"] = float(record.size_mb)
            result[artifact_id]["total_cost"] = total_cost

        return result

    def artifact_audit(self, artifact_type: str, artifact_id: str) -> Optional[List[Dict[str, Any]]]:
        self._require_model_type(artifact_type)
        record = self._models.get(artifact_id)
        if record is None:
            return None

        now = datetime.now(timezone.utc)
        earlier = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        now_iso = now.isoformat().replace("+00:00", "Z")
        metadata = record.to_metadata()

        return [
            {
                "user": {"name": "registry-daemon", "is_admin": True},
                "date": earlier,
                "artifact": metadata,
                "action": "CREATE",
            },
            {
                "user": {"name": "qa-automation", "is_admin": False},
                "date": now_iso,
                "artifact": metadata,
                "action": "RATE",
            },
        ]

    def search_by_name(self, name: str) -> List[Dict[str, Any]]:
        target = str(name or "").strip().lower()
        if not target:
            raise ValueError("name is required.")

        return [
            record.to_metadata()
            for record in self._iter_models()
            if record.name.lower() == target
        ]

    def search_by_regex(self, pattern: str) -> List[Dict[str, Any]]:
        regex_text = str(pattern or "").strip()
        if not regex_text:
            raise ValueError("regex is required.")
        try:
            compiled = re.compile(regex_text, re.IGNORECASE)
        except re.error as exc:
            raise ValueError("Invalid regular expression supplied.") from exc

        matches: List[Dict[str, Any]] = []
        for record in self._iter_models():
            haystack = "\n".join((record.name, record.description, record.card_excerpt))
            if compiled.search(haystack):
                matches.append(record.to_metadata())
        return matches

    def component_health(self, window_minutes: int = 60, include_timeline: bool = False) -> Dict[str, Any]:
        if window_minutes < 5 or window_minutes > 1440:
            raise ValueError("windowMinutes must be between 5 and 1440.")

        stats = self.stats()
        now = datetime.now(timezone.utc)
        observed = now.isoformat().replace("+00:00", "Z")

        components: List[Dict[str, Any]] = [
            {
                "id": "artifact-store",
                "display_name": "Artifact Store",
                "status": "ok",
                "observed_at": observed,
                "description": "Stores artifact metadata and bundles.",
                "issue_count": 0,
                "last_event_at": observed,
                "metrics": {
                    "total_artifacts": stats["total_models"],
                    "vetted": stats["vetted"],
                    "unvetted": stats["unvetted"],
                },
                "issues": [],
                "logs": [
                    {
                        "name": "store.log",
                        "url": "s3://acme-artifacts/logs/store.log",
                    }
                ],
            },
            {
                "id": "evaluation-service",
                "display_name": "Evaluation Service",
                "status": "ok" if stats["vetted"] >= stats["unvetted"] else "degraded",
                "observed_at": observed,
                "description": "Calculates model quality scores.",
                "issue_count": 0,
                "last_event_at": observed,
                "metrics": {
                    "queued_requests": len(self._ingest_requests),
                },
                "issues": [],
                "logs": [
                    {
                        "name": "metrics.log",
                        "url": "s3://acme-artifacts/logs/metrics.log",
                    }
                ],
            },
        ]

        if include_timeline:
            start_bucket = (now - timedelta(minutes=window_minutes)).isoformat().replace("+00:00", "Z")
            end_bucket = now.isoformat().replace("+00:00", "Z")
            for component in components:
                component["timeline"] = [
                    {
                        "bucket": start_bucket,
                        "value": stats["total_models"],
                        "unit": "artifacts",
                    },
                    {
                        "bucket": end_bucket,
                        "value": stats["total_models"],
                        "unit": "artifacts",
                    },
                ]

        return {
            "generated_at": observed,
            "window_minutes": window_minutes,
            "components": components,
        }

    def planned_tracks(self) -> Dict[str, List[str]]:
        return {
            "plannedTracks": [
                "Performance track",
                "Access control track",
                "High assurance track",
            ]
        }

    def simple_license_check(self, artifact_id: str, github_url: str) -> Optional[bool]:
        record = self._models.get(artifact_id)
        if record is None:
            return None
        if not github_url:
            raise ValueError("github_url is required.")

        normalized = record.license_id.lower()
        return normalized in {"apache-2.0", "mit", "bsd-3-clause", "acme-proprietary"}

    def artifact_lineage_graph(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        graph = self.lineage(artifact_id)
        if graph is None:
            return None

        nodes = [
            {
                "artifact_id": node["id"],
                "name": node["name"],
                "source": "registry",
                "metadata": {
                    "license": node.get("license"),
                    "vetted": node.get("vetted"),
                },
            }
            for node in graph["nodes"]
        ]

        edges = [
            {
                "from_node_artifact_id": edge["from"],
                "to_node_artifact_id": edge["to"],
                "relationship": "derived_from",
            }
            for edge in graph["edges"]
        ]

        return {"nodes": nodes, "edges": edges}

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
