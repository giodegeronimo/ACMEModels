#!/usr/bin/env bash
set -euo pipefail

rm -f .coverage.component

rm -f .coverage
pytest -q \
  --cov=src --cov-report=term-missing --cov-fail-under=90 \
  tests \
  --ignore=tests/web \
  --ignore-glob='tests/test_*handler.py' \
  --ignore=tests/test_end_to_end.py

# pytest-cov writes to .coverage by default; rename to the tracked file name.
mv .coverage .coverage.component

coverage report --data-file=.coverage.component -m
coverage html -d htmlcov-component --data-file=.coverage.component
