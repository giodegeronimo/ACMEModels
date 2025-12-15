"""

ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

"""
import os

HTML_BANNER = (
    "<!--\nACMEModels Repository\nIntroductory remarks: "
    "This HTML file is part of the ACMEModels frontend.\n-->\n\n"
)
YAML_BANNER = "#\n# ACMEModels Repository\n# Introductory remarks: This configuration is part of ACMEModels.\n#\n\n"
SHELL_BANNER = "#\n# ACMEModels Repository\n# Introductory remarks: This script is part of ACMEModels.\n#\n\n"

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


def add_header(content: str, banner: str, comment_start: str) -> str:
    """
    add_header: Function description.
    :param content:
    :param banner:
    :param comment_start:
    :returns:
    """

    stripped = content.lstrip()
    if banner.strip() in stripped:
        return content
    # Preserve shebang for shell files
    if content.startswith("#!"):
        idx = content.find("\n")
        return content[: idx + 1] + banner + content[idx + 1:]
    return banner + content


def process_file(path: str, kind: str, apply: bool) -> bool:
    """
    process_file: Function description.
    :param path:
    :param kind:
    :param apply:
    :returns:
    """

    with open(path, "r", encoding="utf-8") as f:
        original = f.read()
    if kind == "html":
        updated = add_header(original, HTML_BANNER, "<!--")
    elif kind in ("yml", "yaml"):
        updated = add_header(original, YAML_BANNER, "#")
    elif kind in ("sh", "bash", "zsh"):
        updated = add_header(original, SHELL_BANNER, "#")
    else:
        return False
    changed = updated != original
    if changed and apply:
        with open(path, "w", encoding="utf-8") as f:
            f.write(updated)
    return changed


def walk(root: str, exts):
    """
    walk: Function description.
    :param root:
    :param exts:
    :returns:
    """

    files = []
    for d, dirnames, fns in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRNAMES]
        for fn in fns:
            for ext in exts:
                if fn.endswith(f".{ext}"):
                    files.append(os.path.join(d, fn))
                    break
    return files


def main():
    """
    main: Function description.
    :param:
    :returns:
    """

    import argparse

    parser = argparse.ArgumentParser(description="Add top-of-file headers to misc files")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--root", default=os.getcwd())
    args = parser.parse_args()

    files = walk(args.root, ["html", "yml", "yaml", "sh", "bash", "zsh"])
    changed = []
    for p in files:
        ext = p.split(".")[-1].lower()
        if process_file(p, ext, args.apply):
            changed.append(p)
    print(f"{'Modified' if args.apply else 'Would modify'} {len(changed)} files.")
    for p in changed:
        print(p)


if __name__ == "__main__":
    main()
