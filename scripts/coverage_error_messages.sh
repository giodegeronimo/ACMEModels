#!/usr/bin/env bash
set -euo pipefail

# Must be run after scripts/coverage_integration.sh (or an equivalent command)
# to produce a `.coverage.integration` file.
python3 scripts/error_raise_coverage.py --data-file .coverage.integration --fail-under 80

