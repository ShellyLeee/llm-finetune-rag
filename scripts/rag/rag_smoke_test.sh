#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CORPUS_PATH="${CORPUS_PATH:-${REPO_ROOT}/data/corpus/raw/wiki_demo.txt}"
INDEX_PATH="${INDEX_PATH:-${REPO_ROOT}/data/corpus/indexes/wiki_demo.faiss}"
MAPPING_PATH="${MAPPING_PATH:-${REPO_ROOT}/data/corpus/chunks/wiki_demo_chunks.json}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-BAAI/bge-small-zh-v1.5}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-64}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TOP_K="${TOP_K:-3}"
QUERY="${1:-什么是监督微调？}"
REPORT_DIR="${REPORT_DIR:-${REPO_ROOT}/reports/latest/rag_smoke}"

mkdir -p "${REPORT_DIR}"
cd "${REPO_ROOT}"

RETRIEVE_JSON="${REPORT_DIR}/retrieval.json"
GENERATE_JSON="${REPORT_DIR}/generation.json"
SUMMARY_JSON="${REPORT_DIR}/summary.json"

echo "[rag_smoke_test] Step 1/4 build index"
python -m src.rag.build_index \
  --corpus-path "${CORPUS_PATH}" \
  --index-path "${INDEX_PATH}" \
  --mapping-path "${MAPPING_PATH}" \
  --embedding-model-name "${EMBEDDING_MODEL_NAME}" \
  --chunk-size "${CHUNK_SIZE}" \
  --chunk-overlap "${CHUNK_OVERLAP}" \
  --batch-size "${BATCH_SIZE}"

echo "[rag_smoke_test] Step 2/4 retrieve top-k"
python -m src.rag.retrieve \
  --query "${QUERY}" \
  --top-k "${TOP_K}" \
  --index-path "${INDEX_PATH}" \
  --mapping-path "${MAPPING_PATH}" \
  --embedding-model-name "${EMBEDDING_MODEL_NAME}" \
  > "${RETRIEVE_JSON}"

echo "[rag_smoke_test] Step 3/4 generate answer"
python -m src.rag.generate \
  --query "${QUERY}" \
  --top-k "${TOP_K}" \
  --index-path "${INDEX_PATH}" \
  --mapping-path "${MAPPING_PATH}" \
  --embedding-model-name "${EMBEDDING_MODEL_NAME}" \
  > "${GENERATE_JSON}"

echo "[rag_smoke_test] Step 4/4 evaluate citation coverage"
python - <<'PY' "${GENERATE_JSON}" "${SUMMARY_JSON}" "${QUERY}" "${TOP_K}" "${INDEX_PATH}" "${MAPPING_PATH}" "${EMBEDDING_MODEL_NAME}" "${CORPUS_PATH}" "${CHUNK_SIZE}" "${CHUNK_OVERLAP}"
import json
import sys
from pathlib import Path

from src.eval.hallucination import compute_faithfulness

gen_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
query = sys.argv[3]
top_k = int(sys.argv[4])
index_path = sys.argv[5]
mapping_path = sys.argv[6]
embedding_model_name = sys.argv[7]
corpus_path = sys.argv[8]
chunk_size = int(sys.argv[9])
chunk_overlap = int(sys.argv[10])

payload = json.loads(gen_path.read_text(encoding="utf-8"))
answer = payload.get("answer", "")
retrieved_chunks = payload.get("retrieved_chunks", [])
metrics = compute_faithfulness(answer, retrieved_chunks)

summary = {
    "smoke_test": "pass" if metrics["covered_chunk_count"] > 0 else "warn",
    "query": query,
    "answer": answer,
    "retrieved_chunk_count": len(retrieved_chunks),
    "config": {
        "corpus_path": corpus_path,
        "index_path": index_path,
        "mapping_path": mapping_path,
        "embedding_model_name": embedding_model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
    },
    "metrics": metrics,
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[rag_smoke_test] Done"
echo "[rag_smoke_test] retrieval: ${RETRIEVE_JSON}"
echo "[rag_smoke_test] generation: ${GENERATE_JSON}"
echo "[rag_smoke_test] summary: ${SUMMARY_JSON}"
