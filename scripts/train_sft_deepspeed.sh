#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train/sft_lora_qwen2.5_7b.yaml}"
# Compatible parameter parsing:
# 1) old usage: train_sft_deepspeed.sh <config> <ds_config>
# 2) new usage: train_sft_deepspeed.sh <config> <run_tag> [ds_config]
ARG2="${2:-}"
ARG3="${3:-}"
RUN_TAG=""
DS_CONFIG="ds_config/zero2.json"
if [[ -n "${ARG2}" ]]; then
  if [[ "${ARG2}" == *.json ]] || [[ -f "${ARG2}" ]]; then
    DS_CONFIG="${ARG2}"
  else
    RUN_TAG="${ARG2}"
    DS_CONFIG="${ARG3:-ds_config/zero2.json}"
  fi
fi
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-$HOME/llm_project/LlamaFactory}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
  echo "[train_sft_deepspeed] CUDA_VISIBLE_DEVICES not set. Defaulting to ${CUDA_VISIBLE_DEVICES}."
else
  echo "[train_sft_deepspeed] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}."
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[train_sft_deepspeed] Config not found: ${CONFIG_PATH}"
  exit 1
fi

if [[ ! -f "${DS_CONFIG}" ]]; then
  echo "[train_sft_deepspeed] DeepSpeed config not found: ${DS_CONFIG}"
  exit 1
fi

if [[ ! -d "${LLAMA_FACTORY_DIR}" ]]; then
  echo "[train_sft_deepspeed] LLAMA_FACTORY_DIR not found: ${LLAMA_FACTORY_DIR}"
  exit 1
fi

if ! command -v deepspeed >/dev/null 2>&1; then
  echo "[train_sft_deepspeed] deepspeed command not found."
  echo "[train_sft_deepspeed] Next step: install requirements/deepspeed.txt in the active environment."
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
  --launcher deepspeed \
  --ds-config "${DS_CONFIG}")"

RUN_NAME="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["run_name"])' <<< "${RUN_META_JSON}")"
MODEL_ALIAS="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["model_alias"])' <<< "${RUN_META_JSON}")"
METHOD="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["method"])' <<< "${RUN_META_JSON}")"
OUTPUT_DIR="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["output_dir"])' <<< "${RUN_META_JSON}")"
TEMP_CONFIG_PATH="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["temp_config_path"])' <<< "${RUN_META_JSON}")"
LAUNCH_TIME="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["launch_time"])' <<< "${RUN_META_JSON}")"
GIT_COMMIT="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["git_commit"])' <<< "${RUN_META_JSON}")"

if command -v llamafactory-cli >/dev/null 2>&1; then
  TRAIN_CMD=(
    deepspeed
    --num_gpus="${GPU_COUNT}"
    llamafactory-cli
    train
    "${TEMP_CONFIG_PATH}"
    --deepspeed
    "${DS_CONFIG}"
  )
else
  TRAIN_CMD=(
    deepspeed
    --num_gpus="${GPU_COUNT}"
    python
    "${LLAMA_FACTORY_DIR}/src/train.py"
    "${TEMP_CONFIG_PATH}"
    --deepspeed
    "${DS_CONFIG}"
  )
fi

printf '%q ' "${TRAIN_CMD[@]}" > "${OUTPUT_DIR}/launch_command.txt"
printf '\n' >> "${OUTPUT_DIR}/launch_command.txt"

TRAIN_LOG="${OUTPUT_DIR}/train_stdout.log"

echo "[train_sft_deepspeed] Config      : ${CONFIG_PATH}"
echo "[train_sft_deepspeed] Temp config : ${TEMP_CONFIG_PATH}"
echo "[train_sft_deepspeed] DS config   : ${DS_CONFIG}"
echo "[train_sft_deepspeed] Run tag     : ${RUN_TAG:-<none>}"
echo "[train_sft_deepspeed] Run name    : ${RUN_NAME}"
echo "[train_sft_deepspeed] Model alias : ${MODEL_ALIAS}"
echo "[train_sft_deepspeed] Method      : ${METHOD}"
echo "[train_sft_deepspeed] Output dir  : ${OUTPUT_DIR}"
echo "[train_sft_deepspeed] GPU count   : ${GPU_COUNT}"

echo "[train_sft_deepspeed] Logging stdout/stderr to ${TRAIN_LOG}"

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
  --launcher "deepspeed" \
  --tag "${RUN_TAG}" \
  --status "${RUN_STATUS}" \
  --temp-config-path "${TEMP_CONFIG_PATH}" \
  --ds-config "${DS_CONFIG}"

exit ${TRAIN_EXIT}
