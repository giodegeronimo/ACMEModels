# outputformatter.py
"""
Minimal, well-commented NDJSON writer you can reuse anywhere.

Key ideas:
- "NDJSON" (aka JSON Lines) = one JSON object per line in a text file.
- We keep output *strict and stable*: numeric score fields can be clamped to [0,1],
  latency fields coerced to non-negative integers (ms), and optional schema validation.
- Absolutely no logs or debug text are written here. Keep machine output clean.

This module has NO project-specific field names. You tell it which keys are scores,
which keys are latencies, and (optionally) give it a JSON schema to enforce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, TextIO
import io
import json
import math

# --- Optional schema validation (safe import) ------------------------------
# If jsonschema isn't installed, validation becomes a no-op.
try:
    from jsonschema import validate as _json_validate  # type: ignore
except Exception:  # pragma: no cover
    _json_validate = None


# --- Generic helpers -------------------------------------------------------

def clamp01(x: Any) -> float:
    """
    Clamp any numeric-ish value into [0,1] with light rounding.
    - Non-numbers (or NaN) become 0.0.
    - Small rounding stabilizes output across runs.
    """
    try:
        f = float(x)
        if math.isnan(f):  # guard against NaN
            return 0.0
        return max(0.0, min(1.0, round(f, 4)))
    except Exception:
        return 0.0


def as_nonneg_int(x: Any) -> int:
    """
    Coerce a value to a non-negative integer.
    Useful for latency fields measured in milliseconds.
    """
    try:
        return max(0, int(x))
    except Exception:
        return 0


# --- The formatter ---------------------------------------------------------

@dataclass
class OutputFormatter:
    """
    A tiny utility that writes one JSON object per line to a given file handle.

    Features:
      - Optional *schema* (dict) validation per line (jsonschema if installed).
      - Optional *score_keys*: keys to clamp into [0,1].
      - Optional *latency_keys*: keys to coerce into non-negative integers.

    Typical usage:
        # 1) easiest: open to a path
        fmt = OutputFormatter.to_path("results.ndjson",
                                      score_keys={"net_score", "license"},
                                      latency_keys={"net_score_latency", "license_latency"},
                                      schema=MY_SCHEMA)

        # 2) write a record (will clamp/validate/flush)
        fmt.write_line({"name": "bert-base-uncased", "net_score": 0.7349, "net_score_latency": 183})

        # 3) close when done (only if we opened the file for you)
        fmt.close()

    Notes:
      - We *never* print logs here. If you need logs, send them to a logger/file in the caller.
      - If you pass your own file handle (fh=...), you own its lifecycle.
    """
    fh: TextIO
    score_keys: Iterable[str] = field(default_factory=set)
    latency_keys: Iterable[str] = field(default_factory=set)
    schema: Optional[Dict[str, Any]] = None
    _owns_handle: bool = False  # True only if we opened the file path for you

    # ------------------------------- Creation ------------------------------

    @classmethod
    def to_path(
        cls,
        path: str,
        *,
        score_keys: Iterable[str] = (),
        latency_keys: Iterable[str] = (),
        schema: Optional[Dict[str, Any]] = None,
        append: bool = False,
    ) -> "OutputFormatter":
        """
        Convenience constructor: open a file at 'path' and return a formatter.
        Use append=True if you want to add to an existing NDJSON.
        """
        mode = "a" if append else "w"
        fh = open(path, mode, encoding="utf-8", newline="\n")
        return cls(
            fh=fh,
            score_keys=set(score_keys),
            latency_keys=set(latency_keys),
            schema=schema,
            _owns_handle=True,
        )

    # ------------------------------- Writing -------------------------------

    def coerce_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply generic safety rules:
        - Clamp score_keys to [0,1]
        - Coerce latency_keys to non-negative ints
        Returns a *new* dict; does not mutate the input.
        """
        out: Dict[str, Any] = dict(record)  # shallow copy

        for k in self.score_keys:
            if k in out:
                out[k] = clamp01(out[k])

        for k in self.latency_keys:
            if k in out:
                out[k] = as_nonneg_int(out[k])

        return out

    def validate_against_schema(self, record: Dict[str, Any]) -> None:
        """
        Validate the dict against the provided JSON schema (if any and if jsonschema is available).
        Raise ValueError with a human-friendly message if validation fails.
        """
        if not self.schema:
            return  # no schema provided -> nothing to validate
        if _json_validate is None:  # jsonschema not installed
            return
        try:
            _json_validate(instance=record, schema=self.schema)
        except Exception as e:
            raise ValueError(f"NDJSON schema validation failed: {e}") from e

    def write_line(self, record: Dict[str, Any]) -> None:
        """
        Write exactly one JSON object as a single line.
        - Coerces/validates (if configured)
        - Always ends with '\n' and flushes (so graders can stream-read)
        """
        safe = self.coerce_fields(record)
        self.validate_against_schema(safe)
        self.fh.write(json.dumps(safe, ensure_ascii=False) + "\n")
        self.fh.flush()

    # ------------------------------- Cleanup -------------------------------

    def close(self) -> None:
        """Close the underlying file ONLY if we opened it (to_path)."""
        if self._owns_handle:
            try:
                self.fh.close()
            except Exception:
                pass

    # Make it usable with "with OutputFormatter.to_path(...)" if you want.
    def __enter__(self) -> "OutputFormatter":  # pragma: no cover
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        self.close()
