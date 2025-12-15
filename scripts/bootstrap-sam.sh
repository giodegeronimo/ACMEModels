#!/usr/bin/env bash
#
# ACMEModels Repository
# Introductory remarks: This script is part of ACMEModels.
#


set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM_DIR="${ROOT_DIR}/backend"
DEFAULT_REGION="us-east-2"

echo "== ACME Models SAM bootstrap =="

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command '$1' not found on PATH" >&2
    exit 1
  fi
}

require_cmd aws
require_cmd sam

echo "AWS CLI: $(aws --version 2>&1)"
echo "SAM CLI: $(sam --version)"

REGION="${AWS_REGION:-$DEFAULT_REGION}"
echo "Using AWS region: ${REGION}"

echo "Checking AWS credentials..."
if ! aws sts get-caller-identity --output text >/dev/null 2>&1; then
  echo "error: unable to get caller identity. Configure AWS credentials before deploying." >&2
  exit 1
fi
echo "AWS credentials look good."

echo "Validating SAM template..."
pushd "${SAM_DIR}" >/dev/null
sam validate --region "${REGION}"
popd >/dev/null

cat <<'EOF'

Next steps:
  1. (Optional) sam build --parallel        # from backend/
  2. sam deploy                             # uses backend/samconfig.toml defaults

Tip: export AWS_REGION if you deploy outside us-east-2.

EOF
