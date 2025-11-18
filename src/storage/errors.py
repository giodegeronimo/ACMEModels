"""Common repository errors used across storage adapters."""

from __future__ import annotations


class RepositoryError(RuntimeError):
    """Base class for storage layer failures."""


class ValidationError(RepositoryError):
    """Raised when user input fails schema validation."""


class ArtifactNotFound(RepositoryError):
    """Raised when a requested artifact id does not exist."""


class LineageNotFound(RepositoryError):
    """Raised when a lineage graph cannot be located."""


class AuditLogNotFound(RepositoryError):
    """Raised when an artifact audit trail has no entries."""
