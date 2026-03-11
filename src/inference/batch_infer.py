from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.inference.generate_answers import (
    generate_single_sample,
    load_model_and_tokenizer,
    save_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch inference and write predictions.jsonl")
    parser.add_argument("--mode", type=str, required=True, choices=["base", "sft", "rag", "sft_rag"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--eval_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument(
        "--index_path",
        type=str,
        default="data/corpus/indexes/wiki_demo.faiss",
        help="Used in rag/sft_rag modes.",
    )
    parser.add_argument(
        "--mapping_path",
        type=str,
        default="data/corpus/chunks/wiki_demo_chunks.json",
        help="Used in rag/sft_rag modes.",
    )
    parser.add_argument("--embedding_model_name", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all samples.")
    return parser.parse_args()


def load_eval_samples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    samples = load_eval_samples(args.eval_file)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    predictions = []
    for sample in samples:
        pred = generate_single_sample(
            sample=sample,
            mode=args.mode,
            model=model,
            tokenizer=tokenizer,
            top_k=args.top_k,
            index_path=args.index_path,
            mapping_path=args.mapping_path,
            embedding_model_name=args.embedding_model_name,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        predictions.append(pred)

    save_jsonl(predictions, args.output_file)
    print(f"[batch_infer] Wrote {len(predictions)} predictions to {args.output_file}")


if __name__ == "__main__":
    main()
