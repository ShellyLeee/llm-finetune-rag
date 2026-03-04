#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install \
  pyyaml \
  pandas \
  jsonlines \
  numpy \
  scikit-learn

echo "[install_deps] Installed local repo dependencies."
echo "[install_deps] TODO: choose faiss-cpu or faiss-gpu when you implement a real retrieval index."
echo "[install_deps] LLaMA-Factory must be installed separately under ~/llm_project/LlamaFactory and used from the matching conda environment."

