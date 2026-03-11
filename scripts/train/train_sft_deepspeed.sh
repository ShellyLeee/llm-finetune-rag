#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_PATH="${1:-configs/train/sft_lora_qwen2.5_7b.yaml}"
ARG2="${2:-}"
ARG3="${3:-}"
RUN_NAME=""
DS_CONFIG="ds_config/zero2.json"

if [[ -n "${ARG2}" ]]; then
  if [[ "${ARG2}" == *.json ]] || [[ -f "${ARG2}" ]]; then
    DS_CONFIG="${ARG2}"
  else
    RUN_NAME="${ARG2}"
    DS_CONFIG="${ARG3:-ds_config/zero2.json}"
  fi
fi

LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-$HOME/llm_project/LlamaFactory}"

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
fi
if [[ "${DS_CONFIG}" != /* ]]; then
  DS_CONFIG="${REPO_ROOT}/${DS_CONFIG}"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0,1,2,3"
  echo "[train_sft_deepspeed] CUDA_VISIBLE_DEVICES not set. Defaulting to ${CUDA_VISIBLE_DEVICES}."
else
  echo "[train_sft_deepspeed] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}."
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[train_sft_deepspeed] Config not found: ${CONFIG_PATH}"
  exit 1
fi

if [[ ! -d "${LLAMA_FACTORY_DIR}" ]]; then
  echo "[train_sft_deepspeed] LLAMA_FACTORY_DIR not found: ${LLAMA_FACTORY_DIR}"
  exit 1
fi

BASE_OUTPUT_DIR="$(python - <<'PY' "${CONFIG_PATH}"
import sys
from pathlib import Path
path = Path(sys.argv[1])
output_dir = "runs/default"
for line in path.read_text(encoding="utf-8").splitlines():
    if line.strip().startswith("output_dir:"):
        output_dir = line.split(":", 1)[1].strip()
        break
print(output_dir)
PY
)"

if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="$(basename "${BASE_OUTPUT_DIR}")"
fi

export RUN_NAME
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_DIR="$(dirname "${BASE_OUTPUT_DIR}")/${RUN_NAME}"
if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${REPO_ROOT}/runs"

GIT_HASH="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
{
  echo "config_path=${CONFIG_PATH}"
  echo "ds_config=${DS_CONFIG}"
  echo "run_name=${RUN_NAME}"
  echo "resolved_output_dir=${OUTPUT_DIR}"
  echo "git_commit=${GIT_HASH}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "llama_factory_dir=${LLAMA_FACTORY_DIR}"
} > "${OUTPUT_DIR}/meta.txt"

echo "[train_sft_deepspeed] Config      : ${CONFIG_PATH}"
echo "[train_sft_deepspeed] DS config   : ${DS_CONFIG}"
echo "[train_sft_deepspeed] Run name    : ${RUN_NAME}"
echo "[train_sft_deepspeed] Output dir  : ${OUTPUT_DIR}"
echo "[train_sft_deepspeed] Launching via llamafactory-cli train (FORCE_TORCHRUN=1)"

llamafactory-cli train "${CONFIG_PATH}"