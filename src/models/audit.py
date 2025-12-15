"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Audit trail domain models.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from .artifacts import ArtifactMetadata


class ArtifactAuditAction(str, Enum):
    """Supported audit actions per spec."""

    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DOWNLOAD = "DOWNLOAD"
    RATE = "RATE"
    AUDIT = "AUDIT"


def _ensure_utc(dt: datetime) -> datetime:
    """
    _ensure_utc: Function description.
    :param dt:
    :returns:
    """

    if dt.tzinfo is None:
        raise ValueError(
            "Audit timestamps must include timezone information (UTC)"
        )
    if dt.tzinfo != timezone.utc:
        return dt.astimezone(timezone.utc)
    return dt


@dataclass(frozen=True)
class User:
    """Minimal user info stored with audit entries."""

    name: str
    is_admin: bool

    def __post_init__(self) -> None:
        """
        __post_init__: Function description.
        :param:
        :returns:
        """

        if not self.name:
            raise ValueError("User name cannot be empty")


@dataclass(frozen=True)
class ArtifactAuditEntry:
    """Single audit record."""

    user: User
    date: datetime
    artifact: ArtifactMetadata
    action: ArtifactAuditAction

    def __post_init__(self) -> None:
        """
        __post_init__: Function description.
        :param:
        :returns:
        """

        object.__setattr__(self, "date", _ensure_utc(self.date))
