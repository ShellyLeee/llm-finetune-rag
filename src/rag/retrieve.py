from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _import_faiss() -> Any:
    try:
        import faiss  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "faiss is required. Install via `pip install faiss-cpu` or your GPU FAISS package."
        ) from exc
    return faiss


def _load_sentence_transformer(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "sentence-transformers is required. Install via `pip install sentence-transformers`."
        ) from exc
    return SentenceTransformer(model_name)


def load_mapping(mapping_path: Path) -> dict[str, Any]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"Chunk mapping file not found: {mapping_path}")
    return json.loads(mapping_path.read_text(encoding="utf-8"))


def retrieve(
    query: str,
    top_k: int = 3,
    index_path: Path | str = Path("data/corpus/indexes/wiki_demo.faiss"),
    mapping_path: Path | str = Path("data/corpus/chunks/wiki_demo_chunks.json"),
    embedding_model_name: str | None = None,
) -> list[dict[str, Any]]:
    index_path = Path(index_path)
    mapping_path = Path(mapping_path)

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    metadata = load_mapping(mapping_path)
    chunks: list[dict[str, Any]] = metadata.get("chunks", [])
    if not chunks:
        return []

    model_name = embedding_model_name or metadata.get("embedding_model_name")
    if not model_name:
        raise ValueError("embedding_model_name not provided and missing from mapping metadata")

    model = _load_sentence_transformer(model_name)
    qvec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    qvec = np.asarray(qvec, dtype=np.float32)

    faiss = _import_faiss()
    index = faiss.read_index(str(index_path))
    k = max(1, min(int(top_k), len(chunks)))
    scores, indices = index.search(qvec, k)

    results: list[dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        item = chunks[idx]
        results.append(
            {
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "score": float(score),
                "rank": rank,
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve top-k chunks from a FAISS index.")
    parser.add_argument("--query", type=str, required=True, help="User query text.")
    parser.add_argument("--top-k", type=int, default=3, help="Top K to retrieve.")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("data/corpus/indexes/wiki_demo.faiss"),
        help="FAISS index path.",
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=Path("data/corpus/chunks/wiki_demo_chunks.json"),
        help="Chunk mapping json path.",
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default=None,
        help="Optional override of embedding model name.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    items = retrieve(
        query=args.query,
        top_k=args.top_k,
        index_path=args.index_path,
        mapping_path=args.mapping_path,
        embedding_model_name=args.embedding_model_name,
    )
    print(json.dumps(items, ensure_ascii=False, indent=2))
