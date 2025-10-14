from __future__ import annotations

"""Base class for metrics used in the ACME Models CLI."""

from abc import ABC, abstractmethod
from typing import Dict, Mapping, Union

MetricOutput = Union[float, Mapping[str, float]]


class Metric(ABC):
    """Abstract base for scoring metrics.

    Each metric exposes a human-friendly ``name`` and an internal ``key``
    identifier. Subclasses must implement :meth:`compute`, returning either a
    float in ``[0, 1]`` or a mapping of floats in that range.
    """

    name: str
    key: str

    def __init__(self, *, name: str, key: str) -> None:
        self.name = name
        self.key = key

    @abstractmethod
    def compute(self, url_record: Dict[str, str]) -> MetricOutput:
        """Produce a metric value for the given URL record."""

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(name={self.name!r}, key={self.key!r})"
