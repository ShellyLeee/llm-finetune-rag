from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path


DEFAULT_TAR_PATH = Path("/mnt/sharedata/ssd_large/common/datasets/wiki-18-corpus/wiki-18.jsonl")
DEFAULT_MEMBER_NAME = "data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl"
DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample passages from tar-packed wiki jsonl.")
    parser.add_argument("--tar_path", type=Path, default=DEFAULT_TAR_PATH)
    parser.add_argument("--member_name", type=str, default=DEFAULT_MEMBER_NAME)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--min_text_len", type=int, default=50)
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL_NAME,
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = text.strip()
    text = text.replace("\x00", " ")
    return " ".join(text.split())


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    current_offset = 0

    with tarfile.open(args.tar_path, "r") as tar:
        member = tar.getmember(args.member_name)
        extracted = tar.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Failed to extract member from tar: {args.member_name}")

        stream = io.TextIOWrapper(extracted, encoding="utf-8")

        for line in stream:
            if len(chunks) >= args.max_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = clean_text(obj.get("contents", ""))
            if len(text) < args.min_text_len:
                continue

            start = current_offset
            end = start + len(text)
            current_offset = end

            chunk_id = f"chunk-{len(chunks):06d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "start": start,
                    "end": end,
                }
            )

    payload = {
        "embedding_model_name": args.embedding_model_name,
        "chunks": chunks,
    }

    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"[prepare_wiki18_chunks_from_tar] Saved {len(chunks)} chunks to {args.output_path}")


if __name__ == "__main__":
    main()