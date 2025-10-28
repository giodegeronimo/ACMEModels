"""Manifest parser for newline-delimited model URL files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

_LOGGER = logging.getLogger(__name__)


class Parser:
    """Parse newline-delimited URL sets into structured dictionaries.

    Each non-empty line corresponds to one model entry. Fields appear in the
    order ``git_url``, ``ds_url``, ``hf_url``. Blank positions keep shorter
    records valid.
    """

    EXPECTED_FIELDS = ("git_url", "ds_url", "hf_url")

    def __init__(self, url_file: Union[Path, str]) -> None:
        self._url_file = Path(url_file)

    def parse(self) -> List[Dict[str, str]]:
        """Convert the parser's source file into a list of URL dictionaries."""
        lines = self._read_lines()

        parsed_records: List[Dict[str, str]] = []
        for line in lines:
            record = self._parse_line(line)
            if record:
                parsed_records.append(record)

        _LOGGER.info(
            "Parsed %d records from %s",
            len(parsed_records),
            self._url_file,
        )

        return parsed_records

    def _read_lines(self) -> List[str]:
        """Load the manifest and preserve ordering for deterministic output."""
        if not self._url_file.exists():
            raise FileNotFoundError(f"URL file not found: {self._url_file}")

        with self._url_file.open("r", encoding="utf-8") as handle:
            lines = [line.rstrip("\n") for line in handle]
        _LOGGER.debug(
            "Read %d raw lines from %s",
            len(lines),
            self._url_file,
        )
        return lines

    def _parse_line(self, line: str) -> Dict[str, str]:
        """Tokenize a single manifest line into the expected URL slots."""
        stripped_line = line.strip()
        if not stripped_line:
            return {}

        parts = [part.strip() for part in stripped_line.split(",")]
        normalized_parts = self._normalize_parts(parts)
        return self._build_record(normalized_parts)

    def _normalize_parts(
        self,
        parts: Sequence[str],
    ) -> Sequence[Optional[str]]:
        """Pad or trim segments so they align with EXPECTED_FIELDS."""
        if len(parts) > len(self.EXPECTED_FIELDS):
            # Ignore extra columns so future additions do not break parsing.
            parts = parts[: len(self.EXPECTED_FIELDS)]
        elif len(parts) < len(self.EXPECTED_FIELDS):
            parts = [*parts, *[""] * (len(self.EXPECTED_FIELDS) - len(parts))]

        return [part or None for part in parts]

    def _build_record(self, parts: Sequence[Optional[str]]) -> Dict[str, str]:
        """Map the normalized segments back onto their field names."""
        record: Dict[str, str] = {}
        for field_name, value in zip(self.EXPECTED_FIELDS, parts):
            if value:
                record[field_name] = value

        return record
