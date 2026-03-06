#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train/sft_lora_qwen2.5_7b.yaml}"
RUN_NAME="${2:-}"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-$HOME/llm_project/LlamaFactory}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
  echo "[train_sft_torchrun] CUDA_VISIBLE_DEVICES not set. Defaulting to ${CUDA_VISIBLE_DEVICES}."
else
  echo "[train_sft_torchrun] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}."
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[train_sft_torchrun] Config not found: ${CONFIG_PATH}"
  exit 1
fi

if [[ ! -d "${LLAMA_FACTORY_DIR}" ]]; then
  echo "[train_sft_torchrun] LLAMA_FACTORY_DIR not found: ${LLAMA_FACTORY_DIR}"
  exit 1
fi

GPU_COUNT="$(python - <<'PY'
import os
gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
print(len(gpus))
PY
)"

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

OUTPUT_DIR="${BASE_OUTPUT_DIR}"
if [[ -n "${RUN_NAME}" ]]; then
  if [[ "${RUN_NAME}" == */* ]]; then
    OUTPUT_DIR="${RUN_NAME}"
  else
    OUTPUT_DIR="$(dirname "${BASE_OUTPUT_DIR}")/${RUN_NAME}"
  fi
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p runs

GIT_HASH="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
{
  echo "config_path=${CONFIG_PATH}"
  echo "launcher=torchrun"
  echo "base_output_dir=${BASE_OUTPUT_DIR}"
  echo "run_name=${RUN_NAME}"
  echo "git_commit=${GIT_HASH}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "llama_factory_dir=${LLAMA_FACTORY_DIR}"
} > "${OUTPUT_DIR}/meta.txt"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export FORCE_TORCHRUN=1

echo "[train_sft_torchrun] Config      : ${CONFIG_PATH}"
echo "[train_sft_torchrun] Run name    : ${RUN_NAME:-<from-config>}"
echo "[train_sft_torchrun] Output dir  : ${OUTPUT_DIR}"
echo "[train_sft_torchrun] GPU count   : ${GPU_COUNT}"

if command -v llamafactory-cli >/dev/null 2>&1; then
  echo "[train_sft_torchrun] Launching via torchrun + llamafactory-cli train"
  torchrun --nproc_per_node="${GPU_COUNT}" --master_port="${MASTER_PORT:-29500}" \
    "$(command -v llamafactory-cli)" train "${CONFIG_PATH}" --output_dir "${OUTPUT_DIR}"
else
  echo "[train_sft_torchrun] llamafactory-cli not found. Falling back to ${LLAMA_FACTORY_DIR}/src/train.py"
  torchrun --nproc_per_node="${GPU_COUNT}" --master_port="${MASTER_PORT:-29500}" \
    "${LLAMA_FACTORY_DIR}/src/train.py" "${CONFIG_PATH}" --output_dir "${OUTPUT_DIR}"
fi
