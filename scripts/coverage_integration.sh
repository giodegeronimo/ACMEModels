#!/usr/bin/env bash
set -euo pipefail

rm -f .coverage.integration

# "Integration/E2E" coverage run: executes the whole tests/ suite.
rm -f .coverage
pytest -q \
  --cov=src --cov-report=term-missing --cov-fail-under=95 \
  tests

# pytest-cov writes to .coverage by default; rename to the tracked file name.
mv .coverage .coverage.integration

coverage report --data-file=.coverage.integration -m
coverage html -d htmlcov-integration --data-file=.coverage.integration
