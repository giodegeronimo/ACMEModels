#!/usr/bin/env bash
set -euo pipefail

rm -f .coverage.integration

# "Integration/E2E" coverage run: executes the whole tests/ suite.
COVERAGE_FILE=.coverage.integration pytest -q \
  --cov=src --cov-report=term-missing --cov-fail-under=95 \
  tests

coverage report --data-file=.coverage.integration -m
coverage html -d htmlcov-integration --data-file=.coverage.integration
