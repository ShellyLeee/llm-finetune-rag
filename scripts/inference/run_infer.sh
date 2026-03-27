#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXP_NAME="default_exp"
MODE="base"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
EVAL_FILE="data/eval/dummy_eval.jsonl"
TOP_K=3
CONFIG_PATH=""
MODE_SET=0
MODEL_PATH_SET=0
EVAL_FILE_SET=0
TOP_K_SET=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config_path)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --exp_name)
      EXP_NAME="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      MODE_SET=1
      shift 2
      ;;
    --model_path)
      MODEL_PATH="$2"
      MODEL_PATH_SET=1
      shift 2
      ;;
    --eval_file)
      EVAL_FILE="$2"
      EVAL_FILE_SET=1
      shift 2
      ;;
    --top_k)
      TOP_K="$2"
      TOP_K_SET=1
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

OUT_DIR="${REPO_ROOT}/reports/experiments/${EXP_NAME}"
OUTPUT_FILE="${OUT_DIR}/predictions.jsonl"
mkdir -p "${OUT_DIR}"

CMD=(
  python -m src.inference.batch_infer
  --output_file "${OUTPUT_FILE}"
)

if [[ -n "${CONFIG_PATH}" ]]; then
  CMD+=(--config_path "${CONFIG_PATH}")
fi

if [[ "${MODE_SET}" -eq 1 || -z "${CONFIG_PATH}" ]]; then
  CMD+=(--mode "${MODE}")
fi
if [[ "${MODEL_PATH_SET}" -eq 1 || -z "${CONFIG_PATH}" ]]; then
  CMD+=(--model_path "${MODEL_PATH}")
fi
if [[ "${EVAL_FILE_SET}" -eq 1 || -z "${CONFIG_PATH}" ]]; then
  CMD+=(--eval_file "${EVAL_FILE}")
fi
if [[ "${TOP_K_SET}" -eq 1 || -z "${CONFIG_PATH}" ]]; then
  CMD+=(--top_k "${TOP_K}")
fi

CMD+=("${EXTRA_ARGS[@]}")
"${CMD[@]}"
