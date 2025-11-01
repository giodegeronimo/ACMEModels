
"""CLI wiring that parses manifests and emits scored NDJSON records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .logging_config import configure_logging
from .metrics.net_score import NetScoreCalculator
from .metrics.registry import MetricDispatcher
from .parser import Parser
from .results import ResultsFormatter, to_ndjson_line


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
        dispatcher = MetricDispatcher()
        metric_results = dispatcher.compute(parsed_urls)
        net_score_calculator = NetScoreCalculator()
        augmented_results = [
            net_score_calculator.with_net_score(results)
            for results in metric_results
        ]
        formatter = ResultsFormatter()
        formatted_records = formatter.format_records(
            parsed_urls,
            augmented_results,
        )
        for record in formatted_records:
            print(to_ndjson_line(record))

        return 0


def handler(event, context):
    """AWS Lambda handler function."""
    # Extract the URL file path from the event
    url_file_path = event.get('url_file_path')
    if not url_file_path:
        raise ValueError("url_file_path is required in the event")

    app = CLIApp(Path(url_file_path))
    return app.run()


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    argument_parser = build_arg_parser()
    parsed_args = argument_parser.parse_args(argv)

    app = CLIApp(parsed_args.url_file)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
