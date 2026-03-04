#!/usr/bin/env bash
set -euo pipefail

echo "[env_check] nvidia-smi"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "[env_check] nvidia-smi not found. Check CUDA driver installation."
fi

echo "[env_check] python version"
python -V

echo "[env_check] pip version"
pip -V

echo "[env_check] LLaMA-Factory CLI"
if command -v llamafactory-cli >/dev/null 2>&1; then
  which llamafactory-cli
  llamafactory-cli -h || true
else
  echo "[env_check] llamafactory-cli not found. Please install LLaMA-Factory first."
fi

echo "[env_check] DeepSpeed"
if command -v deepspeed >/dev/null 2>&1; then
  deepspeed --version || true
else
  echo "[env_check] deepspeed not found. Install LLaMA-Factory requirements/deepspeed.txt in the active environment."
fi

