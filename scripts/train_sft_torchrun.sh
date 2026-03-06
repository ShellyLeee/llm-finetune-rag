#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train/sft_lora_qwen2.5_7b.yaml}"
RUN_TAG="${2:-}"
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

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export FORCE_TORCHRUN=1

RUN_META_JSON="$(python src/utils/run_manager.py prepare \
  --config-path "${CONFIG_PATH}" \
  --tag "${RUN_TAG}" \
  --launcher torchrun)"

RUN_NAME="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["run_name"])' <<< "${RUN_META_JSON}")"
MODEL_ALIAS="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["model_alias"])' <<< "${RUN_META_JSON}")"
METHOD="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["method"])' <<< "${RUN_META_JSON}")"
OUTPUT_DIR="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["output_dir"])' <<< "${RUN_META_JSON}")"
TEMP_CONFIG_PATH="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["temp_config_path"])' <<< "${RUN_META_JSON}")"
LAUNCH_TIME="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["launch_time"])' <<< "${RUN_META_JSON}")"
GIT_COMMIT="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["git_commit"])' <<< "${RUN_META_JSON}")"

if command -v llamafactory-cli >/dev/null 2>&1; then
  TRAIN_CMD=(
    torchrun
    --nproc_per_node="${GPU_COUNT}"
    --master_port="${MASTER_PORT:-29500}"
    "$(command -v llamafactory-cli)"
    train
    "${TEMP_CONFIG_PATH}"
  )
else
  TRAIN_CMD=(
    torchrun
    --nproc_per_node="${GPU_COUNT}"
    --master_port="${MASTER_PORT:-29500}"
    "${LLAMA_FACTORY_DIR}/src/train.py"
    "${TEMP_CONFIG_PATH}"
  )
fi

printf '%q ' "${TRAIN_CMD[@]}" > "${OUTPUT_DIR}/launch_command.txt"
printf '\n' >> "${OUTPUT_DIR}/launch_command.txt"

TRAIN_LOG="${OUTPUT_DIR}/train_stdout.log"

echo "[train_sft_torchrun] Config      : ${CONFIG_PATH}"
echo "[train_sft_torchrun] Temp config : ${TEMP_CONFIG_PATH}"
echo "[train_sft_torchrun] Run tag     : ${RUN_TAG:-<none>}"
echo "[train_sft_torchrun] Run name    : ${RUN_NAME}"
echo "[train_sft_torchrun] Model alias : ${MODEL_ALIAS}"
echo "[train_sft_torchrun] Method      : ${METHOD}"
echo "[train_sft_torchrun] Output dir  : ${OUTPUT_DIR}"
echo "[train_sft_torchrun] GPU count   : ${GPU_COUNT}"

echo "[train_sft_torchrun] Logging stdout/stderr to ${TRAIN_LOG}"

set +e
"${TRAIN_CMD[@]}" 2>&1 | tee "${TRAIN_LOG}"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

RUN_STATUS="success"
if [[ ${TRAIN_EXIT} -ne 0 ]]; then
  RUN_STATUS="failed"
fi

python src/utils/run_manager.py append-index \
  --run-name "${RUN_NAME}" \
  --model-alias "${MODEL_ALIAS}" \
  --method "${METHOD}" \
  --config-path "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --launch-time "${LAUNCH_TIME}" \
  --git-commit "${GIT_COMMIT}" \
  --launcher "torchrun" \
  --tag "${RUN_TAG}" \
  --status "${RUN_STATUS}" \
  --temp-config-path "${TEMP_CONFIG_PATH}"

exit ${TRAIN_EXIT}
