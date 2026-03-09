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


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[dict[str, Any]] = []
    step = chunk_size - chunk_overlap
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(
                {
                    "chunk_id": f"chunk-{idx:06d}",
                    "text": chunk,
                    "start": start,
                    "end": end,
                }
            )
            idx += 1
        if end >= len(text):
            break
        start += step

    return chunks


def encode_texts(
    texts: list[str],
    embedding_model_name: str,
    batch_size: int = 32,
) -> np.ndarray:
    model = _load_sentence_transformer(embedding_model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def build_faiss_index(vectors: np.ndarray) -> Any:
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got shape={vectors.shape}")
    dim = vectors.shape[1]
    faiss = _import_faiss()
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def build_and_save_index(
    corpus_path: Path,
    index_path: Path,
    mapping_path: Path,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
) -> dict[str, Any]:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    text = corpus_path.read_text(encoding="utf-8")
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise RuntimeError(f"No chunks produced from corpus: {corpus_path}")

    vectors = encode_texts(
        [chunk["text"] for chunk in chunks],
        embedding_model_name=embedding_model_name,
        batch_size=batch_size,
    )
    index = build_faiss_index(vectors)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    faiss = _import_faiss()
    faiss.write_index(index, str(index_path))
    mapping_path.write_text(
        json.dumps(
            {
                "embedding_model_name": embedding_model_name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "num_chunks": len(chunks),
                "chunks": chunks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "corpus_path": str(corpus_path),
        "index_path": str(index_path),
        "mapping_path": str(mapping_path),
        "num_chunks": len(chunks),
        "embedding_dim": int(vectors.shape[1]),
        "embedding_model_name": embedding_model_name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index for smoke-test RAG.")
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("data/corpus/raw/wiki_demo.txt"),
        help="Raw text corpus path.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("data/corpus/indexes/wiki_demo.faiss"),
        help="Output FAISS index path.",
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=Path("data/corpus/chunks/wiki_demo_chunks.json"),
        help="Output chunk mapping json path.",
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default="BAAI/bge-small-zh-v1.5",
        help="SentenceTransformer model name/path.",
    )
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Chunk overlap in characters.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_and_save_index(
        corpus_path=args.corpus_path,
        index_path=args.index_path,
        mapping_path=args.mapping_path,
        embedding_model_name=args.embedding_model_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
