from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Any


DEFAULT_TAR_PATH = Path("/mnt/sharedata/ssd_large/common/datasets/wiki-18-corpus/wiki-18.jsonl")
DEFAULT_MEMBER_NAME = "data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl"
DEFAULT_OUTPUT_DIR = Path("/mnt/sharedata/ssd_large/users/liyx/corpus/chunks")
DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample wiki passages from a tar-packed jsonl corpus and save as chunks json."
    )
    parser.add_argument(
        "--tar-path",
        type=Path,
        default=DEFAULT_TAR_PATH,
        help="Path to tar archive containing wiki jsonl.",
    )
    parser.add_argument(
        "--member-name",
        type=str,
        default=DEFAULT_MEMBER_NAME,
        help="Member path inside the tar archive.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output chunks json path. If omitted, auto-generated from sample size.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output chunks json if output-path is not specified.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Number of passages to sample from the corpus.",
    )
    parser.add_argument(
        "--min-text-len",
        type=int,
        default=50,
        help="Skip passages shorter than this number of characters after cleaning.",
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL_NAME,
        help="Embedding model name to store in output metadata.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = text.strip()
    text = text.replace("\x00", " ")
    return " ".join(text.split())


def infer_output_path(output_path: Path | None, output_dir: Path, max_samples: int) -> Path:
    if output_path is not None:
        return output_path
    return output_dir / f"wiki18_{max_samples // 1000}k_chunks.json"


def make_chunk_record(idx: int, text: str, start: int, end: int) -> dict[str, Any]:
    return {
        "chunk_id": f"chunk-{idx:06d}",
        "text": text,
        "start": start,
        "end": end,
    }


def main() -> None:
    args = parse_args()

    output_path = infer_output_path(args.output_path, args.output_dir, args.max_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks: list[dict[str, Any]] = []
    current_offset = 0
    total_seen = 0
    total_kept = 0

    with tarfile.open(args.tar_path, "r") as tar:
        member = tar.getmember(args.member_name)
        extracted = tar.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Failed to extract member from tar: {args.member_name}")

        stream = io.TextIOWrapper(extracted, encoding="utf-8")

        for line in stream:
            total_seen += 1
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # tolerate malformed header / weird line if any
                continue

            text = clean_text(str(obj.get("contents", "")))
            if len(text) < args.min_text_len:
                continue

            start = current_offset
            end = start + len(text)
            current_offset = end

            chunk = make_chunk_record(
                idx=len(chunks),
                text=text,
                start=start,
                end=end,
            )
            chunks.append(chunk)
            total_kept += 1

            if total_kept >= args.max_samples:
                break

    payload = {
        "embedding_model_name": args.embedding_model_name,
        "num_chunks": len(chunks),
        "source": {
            "tar_path": str(args.tar_path),
            "member_name": args.member_name,
            "max_samples": args.max_samples,
            "min_text_len": args.min_text_len,
            "total_lines_seen": total_seen,
        },
        "chunks": chunks,
    }

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "num_chunks": len(chunks),
                "embedding_model_name": args.embedding_model_name,
                "source_tar": str(args.tar_path),
                "source_member": args.member_name,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()