#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <predictions_jsonl> [output_dir]"
  exit 1
fi

PREDICTIONS_FILE="$1"
OUTPUT_DIR="${2:-$(dirname "${PREDICTIONS_FILE}")}"

python -m src.eval.run_eval \
  --predictions_file "${PREDICTIONS_FILE}" \
  --output_dir "${OUTPUT_DIR}"
