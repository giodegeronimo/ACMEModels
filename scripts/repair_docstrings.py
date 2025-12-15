#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

SKIP_DIRNAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "build",
    "dist",
    "node_modules",
    ".aws-sam",
}

_MARKERS = ("Function description.", "Class description.")


def _iter_python_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRNAMES]
        for filename in filenames:
            if filename.endswith(".py"):
                paths.append(Path(dirpath) / filename)
    return paths


def _find_triple_quote_end(lines: list[str], start: int) -> int | None:
    if lines[start].count('"""') >= 2:
        return start
    for idx in range(start + 1, len(lines)):
        if lines[idx].lstrip().startswith('"""'):
            return idx
    return None


def _block_contains_marker(block: list[str]) -> bool:
    joined = "\n".join(block)
    return any(marker in joined for marker in _MARKERS)


def _remove_decorator_adjacent_docstrings(lines: list[str]) -> bool:
    changed = False
    idx = 0
    while idx < len(lines) - 2:
        line = lines[idx]
        next_line = lines[idx + 1]
        if line.lstrip().startswith("@") and next_line.lstrip().startswith('"""'):
            end = _find_triple_quote_end(lines, idx + 1)
            if end is None:
                idx += 1
                continue
            block = lines[idx + 1:end + 1]
            after_idx = end + 1
            while after_idx < len(lines) and not lines[after_idx].strip():
                after_idx += 1
            after = lines[after_idx] if after_idx < len(lines) else ""
            indent = line[: len(line) - len(line.lstrip())]
            if (
                _block_contains_marker(block)
                and after.startswith(indent)
                and after[len(indent):].lstrip().startswith(("def ", "class "))
            ):
                del lines[idx + 1:end + 1]
                if idx + 1 < len(lines) and not lines[idx + 1].strip():
                    del lines[idx + 1]
                changed = True
                continue
        idx += 1
    return changed


def _remove_signature_docstrings(lines: list[str]) -> bool:
    changed = False
    idx = 0
    while idx < len(lines) - 1:
        line = lines[idx]
        if not line.lstrip().startswith('"""'):
            idx += 1
            continue
        end = _find_triple_quote_end(lines, idx)
        if end is None:
            idx += 1
            continue
        block = lines[idx:end + 1]
        next_idx = end + 1
        while next_idx < len(lines) and not lines[next_idx].strip():
            next_idx += 1
        next_line = lines[next_idx] if next_idx < len(lines) else ""
        if _block_contains_marker(block) and next_line.lstrip().startswith(")"):
            if ": ..." in next_line or next_line.rstrip().endswith(": ..."):
                del lines[idx:end + 1]
                if idx < len(lines) and not lines[idx].strip():
                    del lines[idx]
                changed = True
                continue
        idx = end + 1
    return changed


def _ensure_trailing_newline(text: str) -> str:
    if not text.endswith("\n"):
        return text + "\n"
    return text


def _remove_stray_module_strings(lines: list[str]) -> bool:
    if not lines:
        return False

    changed = False

    idx = 0
    if lines[0].lstrip().startswith('"""'):
        end = _find_triple_quote_end(lines, 0)
        if end is None:
            return False
        idx = end + 1

    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    if idx < len(lines) and lines[idx].startswith("from __future__ import "):
        idx += 1

    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    if idx < len(lines) and lines[idx].lstrip().startswith('"""'):
        end = _find_triple_quote_end(lines, idx)
        if end is None:
            return changed
        del lines[idx:end + 1]
        while idx < len(lines) and not lines[idx].strip():
            del lines[idx]
        changed = True

    return changed


def repair_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines()
    changed = False

    if _remove_stray_module_strings(lines):
        changed = True
    if _remove_decorator_adjacent_docstrings(lines):
        changed = True
    if _remove_signature_docstrings(lines):
        changed = True

    updated = "\n".join(lines)
    updated = _ensure_trailing_newline(updated)
    if updated != original:
        changed = True
        path.write_text(updated, encoding="utf-8")
    return changed


def main() -> None:
    root = Path(".")
    touched = 0
    for path in _iter_python_files(root):
        try:
            if repair_file(path):
                touched += 1
        except Exception:
            continue
    print(f"Repaired {touched} Python files.")


if __name__ == "__main__":
    main()
