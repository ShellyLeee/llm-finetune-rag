#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_PATH="${1:-}"
RUN_NAME="${2:-}"
CUDA_LIST="${3:-}"

if [[ -z "${CONFIG_PATH}" || -z "${RUN_NAME}" ]]; then
  echo "Usage: $0 <config_path> <run_name> [cuda_visible_devices]"
  exit 1
fi

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[train_sft] Config not found: ${CONFIG_PATH}"
  exit 1
fi

if [[ -n "${CUDA_LIST}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_LIST}"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0,1,2,3"
fi

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "[train_sft] llamafactory-cli not found in PATH."
  exit 1
fi

# Expected config behavior:
# - deepspeed should be configured in YAML if needed.
# - output_dir/run naming should be handled by YAML via ${RUN_NAME}.
export RUN_NAME="${RUN_NAME}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"

echo "[train_sft] Config: ${CONFIG_PATH}"
echo "[train_sft] RUN_NAME: ${RUN_NAME}"
echo "[train_sft] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "[train_sft] FORCE_TORCHRUN: ${FORCE_TORCHRUN}"

llamafactory-cli train "${CONFIG_PATH}"
