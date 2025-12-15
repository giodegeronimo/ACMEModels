"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Domain models for artifacts and related helpers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

ARTIFACT_ID_REGEX = re.compile(r"^[a-zA-Z0-9\-]+$")


class ArtifactType(str, Enum):
    """Artifact category per the OpenAPI schema."""

    MODEL = "model"
    DATASET = "dataset"
    CODE = "code"


def validate_artifact_id(value: str) -> str:
    """Ensure artifact IDs match the required regex."""
    if not value:
        raise ValueError("Artifact ID cannot be empty")
    if not ARTIFACT_ID_REGEX.match(value):
        raise ValueError(
            "Artifact ID "
            f"'{value}' is invalid. Expected pattern "
            f"{ARTIFACT_ID_REGEX.pattern}"
        )
    return value


def validate_artifact_name(name: str, *, allow_wildcard: bool = False) -> str:
    """
    Validate artifact names.

    The spec reserves '*' for enumerate semantics, so only allow it when
    explicitly requested.
    """
    if not name:
        raise ValueError("Artifact name cannot be empty")
    if name == "*" and not allow_wildcard:
        raise ValueError(
            "Wildcard name '*' is only allowed for enumerate operations"
        )
    return name


def validate_url(url: str) -> str:
    """Minimal URI validation matching the ArtifactData schema guidance."""
    parsed = urlparse(url)
    if not (parsed.scheme and parsed.netloc):
        raise ValueError(
            f"Artifact source URL '{url}' is not a valid absolute URI"
        )
    return url


@dataclass(frozen=True)
class ArtifactData:
    """Source location for ingesting an artifact."""

    url: str

    def __post_init__(self) -> None:
        # Ensure the ingest URL is a valid absolute URI.
        """
        __post_init__: Function description.
        :param:
        :returns:
        """

        validate_url(self.url)


@dataclass(frozen=True)
class ArtifactMetadata:
    """Metadata envelope provided in all artifact APIs."""

    name: str
    id: str
    type: ArtifactType

    def __post_init__(self) -> None:
        """
        __post_init__: Function description.
        :param:
        :returns:
        """

        validate_artifact_name(self.name, allow_wildcard=False)
        validate_artifact_id(self.id)
        if not isinstance(self.type, ArtifactType):
            raise ValueError(
                f"Artifact type '{self.type}' is not recognized"
            )


@dataclass(frozen=True)
class Artifact:
    """Full artifact representation containing metadata and ingest data."""

    metadata: ArtifactMetadata
    data: ArtifactData


@dataclass(frozen=True)
class ArtifactQuery:
    """Query payload used for enumerate operations."""

    name: str
    types: Optional[list[ArtifactType]] = None

    def __post_init__(self) -> None:
        """
        __post_init__: Function description.
        :param:
        :returns:
        """

        validate_artifact_name(self.name, allow_wildcard=True)
        if self.types:
            for t in self.types:
                if not isinstance(t, ArtifactType):
                    raise ValueError(
                        f"Artifact type '{t}' in query filter is invalid"
                    )
