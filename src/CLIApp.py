from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .clients.git_client import GitClient
from .parser import Parser


class CLIApp:
    """Command-line entry point for the ACME Models tool.

    The app currently wires the manifest parser to stdout for rapid iteration.
    Later phases will replace this with the full scoring pipeline.
    """

    def __init__(self, url_file: Path) -> None:
        self._url_file = url_file

    def run(self) -> int:
        # Parse manifest first so errors surface quickly.
        parser = Parser(self._url_file)
        parsed_urls = parser.parse()
        print(json.dumps(parsed_urls, indent=2))

        first_git = next(
            (entry for entry in parsed_urls if "git_url" in entry),
            None,
        )
        if first_git:
            git_client = GitClient()
            try:
                metadata = git_client.get_repo_metadata(first_git["git_url"])
            except Exception as error:  # pragma: no cover - demo path only
                metadata = {"error": str(error)}
            print(
                json.dumps(
                    {"demo_git_metadata": metadata},
                    indent=2,
                    default=str,
                )
            )
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description=(
            "ACME Models CLI: Parse URL manifests "
            "into structured data."
        )
    )
    argument_parser.add_argument(
        "url_file",
        type=Path,
        help="Path to the newline-delimited URL manifest file.",
    )
    return argument_parser


def main(argv: Sequence[str] | None = None) -> int:
    argument_parser = build_arg_parser()
    parsed_args = argument_parser.parse_args(argv)

    app = CLIApp(parsed_args.url_file)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
