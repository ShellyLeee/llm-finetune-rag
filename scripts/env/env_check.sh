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
  echo "[env_check] LLaMA-Factory version"
  python -m pip show llamafactory llmtuner 2>/dev/null | awk '/^Name:|^Version:/{print}'
  echo "[env_check] Native early stopping support (early_stopping_steps)"
  python - <<'PY'
import importlib.util
from pathlib import Path

def check(pkg: str) -> tuple[bool, str]:
    spec = importlib.util.find_spec(pkg)
    if spec is None or spec.origin is None:
        return False, ""
    root = Path(spec.origin).resolve().parent
    tuner = root / "train" / "tuner.py"
    if not tuner.exists():
        return False, str(tuner)
    text = tuner.read_text(encoding="utf-8", errors="ignore")
    return ("early_stopping_steps" in text and "EarlyStoppingCallback" in text), str(tuner)

for candidate in ("llamafactory", "llmtuner"):
    ok, path = check(candidate)
    if path:
        print(f"{candidate}: {'supported' if ok else 'not found in tuner.py'} ({path})")
PY
else
  echo "[env_check] llamafactory-cli not found. Please install LLaMA-Factory first."
fi

echo "[env_check] DeepSpeed"
if command -v deepspeed >/dev/null 2>&1; then
  deepspeed --version || true
else
  echo "[env_check] deepspeed not found. Install LLaMA-Factory requirements/deepspeed.txt in the active environment."
fi
