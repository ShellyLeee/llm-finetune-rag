#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/configs/export/merge_lora.yaml}"

echo "[merge_lora] Placeholder script."
echo "[merge_lora] Config: ${CONFIG_PATH}"
echo "[merge_lora] TODO: wire this to your merge/export backend (e.g., llamafactory-cli export)."
