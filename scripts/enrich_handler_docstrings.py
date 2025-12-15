#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _DocstringBlock:
    indent: str
    start_idx: int
    end_idx: int
    text: str


_DEF_RE = re.compile(r"^(?P<indent>\s*)def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
_TRIPLE_RE = re.compile(r'^(?P<indent>\s*)"""')


def _iter_py_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".aws-sam", "__pycache__"}]
        for filename in filenames:
            if filename.endswith(".py"):
                paths.append(Path(dirpath) / filename)
    return paths


def _find_docstring_block(lines: list[str], def_line_idx: int) -> _DocstringBlock | None:
    indent_match = _DEF_RE.match(lines[def_line_idx])
    if not indent_match:
        return None
    indent = indent_match.group("indent") + " " * 4

    signature_end = def_line_idx
    paren_balance = lines[def_line_idx].count("(") - lines[def_line_idx].count(")")
    scan_idx = def_line_idx
    while scan_idx + 1 < len(lines):
        line = lines[scan_idx].rstrip()
        if paren_balance <= 0 and line.endswith(":"):
            signature_end = scan_idx
            break
        scan_idx += 1
        paren_balance += lines[scan_idx].count("(") - lines[scan_idx].count(")")
    else:
        signature_end = scan_idx

    idx = signature_end + 1
    while idx < len(lines) and (
        not lines[idx].strip() or lines[idx].lstrip().startswith("#")
    ):
        idx += 1
    if idx >= len(lines):
        return None

    triple_match = _TRIPLE_RE.match(lines[idx])
    if not triple_match or triple_match.group("indent") != indent:
        return None

    start_idx = idx
    end_idx = idx
    if lines[idx].count('"""') >= 2:
        end_idx = idx
    else:
        end_idx = idx + 1
        while end_idx < len(lines):
            if lines[end_idx].startswith(indent) and '"""' in lines[end_idx]:
                break
            end_idx += 1
        if end_idx >= len(lines):
            return None

    text = "\n".join(lines[start_idx:end_idx + 1])
    return _DocstringBlock(indent=indent, start_idx=start_idx, end_idx=end_idx, text=text)


def _is_placeholder_docstring(text: str) -> bool:
    return "Function description." in text or "Class description." in text


def _summarize(name: str) -> str:
    if name == "lambda_handler":
        return "AWS Lambda handler entry point."
    if name.startswith("_parse_"):
        return f"Parse and validate `{name.removeprefix('_parse_')}` from the request."
    if name == "_require_auth":
        return "Enforce request authentication for this handler."
    if name == "_extract_auth_token":
        return "Extract an authentication token from the request."
    if name == "_json_response":
        return "Create a JSON API Gateway proxy response."
    if name == "_error_response":
        return "Create a JSON error response payload."
    if name.endswith("_response"):
        return "Create an API Gateway proxy response payload."
    if name.startswith("_decode_"):
        return "Decode and validate request payload data."
    if name.startswith("_encode_"):
        return "Encode a continuation token for pagination."
    if name.startswith("_build_"):
        return "Build a derived URL or response value."
    if name.startswith("_resolve_"):
        return "Resolve configuration from environment/request context."
    if name.startswith("_store_"):
        return "Persist data to a backing store."
    if name.startswith("_load_"):
        return "Load data from a backing store."
    if name.startswith("_validate_"):
        return "Validate request inputs against stored state."
    if name.startswith("_serialize_"):
        return "Serialize a domain object into a JSON payload."
    if name.startswith("_collect_"):
        return "Collect and paginate results from a backing index."
    if name.startswith("_entry_"):
        return "Apply matching logic to a single index entry."
    if name.startswith("_run_with_timeout"):
        return "Execute a callable with a wall-clock timeout."
    return "Helper function."


def _extract_params_from_placeholder(text: str) -> list[str]:
    params: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(":param "):
            name = line.removeprefix(":param ").split(":", 1)[0].strip()
            if name:
                params.append(name)
    return params


def _build_docstring(indent: str, name: str, params: list[str]) -> str:
    summary = _summarize(name)
    lines = ['"""' + summary]
    if params:
        lines.append("")
        for param in params:
            lines.append(f":param {param}:")
    lines.append(":returns:")
    lines.append('"""')
    return "\n".join(indent + line if line else "" for line in lines)


def enrich_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines()
    changed = False

    idx = 0
    while idx < len(lines):
        match = _DEF_RE.match(lines[idx])
        if not match:
            idx += 1
            continue
        name = match.group("name")
        block = _find_docstring_block(lines, idx)
        if block is None:
            idx += 1
            continue
        if _is_placeholder_docstring(block.text):
            params = _extract_params_from_placeholder(block.text)
            replacement = _build_docstring(block.indent, name, params).splitlines()
            lines[block.start_idx:block.end_idx + 1] = replacement
            changed = True
            idx = block.start_idx + len(replacement)
            continue
        idx = block.end_idx + 1

    updated = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
    if updated != original:
        path.write_text(updated, encoding="utf-8")
        return True
    return changed


def main() -> None:
    root = Path("backend/src/handlers")
    touched = 0
    for path in _iter_py_files(root):
        if enrich_file(path):
            touched += 1
    print(f"Enriched docstrings in {touched} files.")


if __name__ == "__main__":
    main()
