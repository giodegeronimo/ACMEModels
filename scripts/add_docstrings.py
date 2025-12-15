"""

ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

"""
import ast
import os
from dataclasses import dataclass
from typing import List, Tuple

INTRO_BANNER = (
    """\nACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.\n"""
)

_SKIP_DIRNAMES = {
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


@dataclass(frozen=True)
class _DocstringSpan:
    """
    _DocstringSpan: Class description.
    """

    start_line: int
    end_line: int
    value: str


def _get_module_docstring_span(tree: ast.Module) -> _DocstringSpan | None:
    """
    _get_module_docstring_span: Function description.
    :param tree:
    :returns:
    """

    if not tree.body:
        return None
    first = tree.body[0]
    if not isinstance(first, ast.Expr):
        return None
    value = getattr(first, "value", None)
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return None
    start = getattr(first, "lineno", None)
    end = getattr(first, "end_lineno", None)
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    return _DocstringSpan(start_line=start, end_line=end, value=value.value)


def _has_intro_remarks(docstring: str) -> bool:
    """
    _has_intro_remarks: Function description.
    :param docstring:
    :returns:
    """

    return "Introductory remarks:" in docstring


def ensure_module_intro_remarks(source: str) -> str:
    """
    ensure_module_intro_remarks: Function description.
    :param source:
    :returns:
    """

    tree = ast.parse(source)
    span = _get_module_docstring_span(tree)
    if span is None:
        return source
    if _has_intro_remarks(span.value):
        return source

    existing = span.value.strip()
    banner = INTRO_BANNER.strip()
    new_body = banner if not existing else f"{banner}\n\n{existing}"
    replacement = f"\"\"\"\n{new_body}\n\"\"\""

    lines = source.splitlines()
    lines[span.start_line - 1:span.end_line] = replacement.splitlines()
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")


def has_module_docstring(tree: ast.AST) -> bool:
    """
    has_module_docstring: Function description.
    :param tree:
    :returns:
    """

    return bool(ast.get_docstring(tree))


def insert_module_docstring(source: str) -> str:
    """
    insert_module_docstring: Function description.
    :param source:
    :returns:
    """

    lines = source.splitlines()
    # Preserve shebang and encoding if present
    insert_idx = 0
    banner = f"\"\"\"\n{INTRO_BANNER}\n\"\"\"\n\n"
    if lines and lines[0].startswith("#!"):
        insert_idx = 1
    # PEP 263 encoding cookie
    if len(lines) > insert_idx and "coding:" in lines[insert_idx]:
        insert_idx += 1
    lines.insert(insert_idx, banner.rstrip("\n"))
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")


def has_docstring(node: ast.AST) -> bool:
    """
    has_docstring: Function description.
    :param node:
    :returns:
    """

    return bool(ast.get_docstring(node))


def generate_docstring_skeleton(node: ast.AST, source_lines: List[str]) -> Tuple[int, str]:
    """
    generate_docstring_skeleton: Function description.
    :param node:
    :param source_lines:
    :returns:
    """

    name = getattr(node, "name", "")
    if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
        # Build parameters list
        params = []
        for arg in node.args.args:
            if arg.arg != "self":
                params.append(arg.arg)
        if node.args.vararg:
            params.append(f"*{node.args.vararg.arg}")
        for kw in node.args.kwonlyargs:
            params.append(kw.arg)
        if node.args.kwarg:
            params.append(f"**{node.args.kwarg.arg}")
        params_section = "\n".join([f":param {p}:" for p in params]) if params else ":param:"
        returns_section = ":returns:"
        header = f"\"\"\"\n{name}: Function description.\n{params_section}\n{returns_section}\n\"\"\""
    elif isinstance(node, ast.ClassDef):
        header = f"\"\"\"\n{name}: Class description.\n\"\"\""
    else:
        header = "\"\"\"\nDescription.\n\"\"\""

    # Determine insertion line (the line after the definition header)
    # ast gives lineno as 1-based line number for the first token of the node
    insert_line = node.body[0].lineno if node.body else node.lineno + 1
    # If first body element is a docstring (Str/Constant), we would have detected has_docstring before
    # Insert the docstring at insert_line - 1 index
    return insert_line - 1, header


def add_missing_docstrings_to_source(source: str) -> str:
    """
    add_missing_docstrings_to_source: Function description.
    :param source:
    :returns:
    """

    tree = ast.parse(source)

    # Module docstring
    if not has_module_docstring(tree):
        source = insert_module_docstring(source)
        tree = ast.parse(source)  # reparse due to changed line numbers
    else:
        updated = ensure_module_intro_remarks(source)
        if updated != source:
            source = updated
            tree = ast.parse(source)

    # Collect insertions for classes and functions
    insertions: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not has_docstring(node):
                idx, doc = generate_docstring_skeleton(node, source.splitlines())
                insertions.append((idx, doc))

    if not insertions:
        return source

    # Apply insertions from bottom to top so indices are stable
    lines = source.splitlines()
    for idx, doc in sorted(insertions, key=lambda x: x[0], reverse=True):
        indent = ""
        # Determine indentation from the next line if present
        if idx < len(lines):
            leading = lines[idx]
            indent = leading[: len(leading) - len(leading.lstrip())]
        doc_lines = [
            indent + line if line else line for line in doc.splitlines()
        ]
        lines[idx:idx] = doc_lines + [""]

    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")


def process_file(path: str, apply: bool) -> bool:
    """
    process_file: Function description.
    :param path:
    :param apply:
    :returns:
    """

    try:
        with open(path, "r", encoding="utf-8") as f:
            original = f.read()
        updated = add_missing_docstrings_to_source(original)
        if updated != original and apply:
            with open(path, "w", encoding="utf-8") as f:
                f.write(updated)
            return True
        return updated != original
    except Exception:
        return False


def find_python_files(root: str) -> List[str]:
    """
    find_python_files: Function description.
    :param root:
    :returns:
    """

    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRNAMES]
        for fn in filenames:
            if fn.endswith(".py"):
                files.append(os.path.join(dirpath, fn))
    return files


def main():
    """
    main: Function description.
    :param:
    :returns:
    """

    import argparse

    parser = argparse.ArgumentParser(description="Add module and member docstrings")
    parser.add_argument("--apply", action="store_true", help="Apply changes to files")
    parser.add_argument(
        "--root",
        default=os.getcwd(),
        help="Root directory to scan (default: current working directory)",
    )
    args = parser.parse_args()

    py_files = find_python_files(args.root)
    changed: List[str] = []
    for path in py_files:
        # Skip __pycache__ or hidden folders
        if "__pycache__" in path:
            continue
        if process_file(path, apply=args.apply):
            changed.append(path)

    action = "Modified" if args.apply else "Would modify"
    print(f"{action} {len(changed)} Python files.")
    for p in changed:
        print(p)


if __name__ == "__main__":
    main()
