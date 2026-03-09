#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-Qwen/Qwen2.5-7B-Instruct}"

if command -v vllm >/dev/null 2>&1; then
  echo "[serve_vllm] vLLM detected."
  echo "[serve_vllm] Example:"
  echo "python -m vllm.entrypoints.openai.api_server --model ${MODEL_PATH} --port 8000"
else
  echo "[serve_vllm] vllm is not installed."
  echo "[serve_vllm] Install it first, then launch an OpenAI-compatible server with:"
  echo "python -m vllm.entrypoints.openai.api_server --model ${MODEL_PATH} --port 8000"
fi

