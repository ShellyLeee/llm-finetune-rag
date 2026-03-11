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
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp_name)
      EXP_NAME="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --eval_file)
      EVAL_FILE="$2"
      shift 2
      ;;
    --top_k)
      TOP_K="$2"
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

python -m src.inference.batch_infer \
  --mode "${MODE}" \
  --model_path "${MODEL_PATH}" \
  --eval_file "${EVAL_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --top_k "${TOP_K}" \
  "${EXTRA_ARGS[@]}"
