#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install \
  pyyaml \
  pandas \
  jsonlines \
  numpy \
  scikit-learn \
  sentence-transformers \
  faiss-cpu

echo "[install_deps] Installed local repo dependencies."
echo "[install_deps] Installed RAG deps: sentence-transformers + faiss-cpu."
echo "[install_deps] If your server uses CUDA FAISS, replace faiss-cpu with a GPU build in your environment."
echo "[install_deps] LLaMA-Factory must be installed separately under ~/llm_project/LlamaFactory and used from the matching conda environment."
