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
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config_path", type=Path, default=None)
    partial_args, _ = bootstrap.parse_known_args()
    config_defaults = load_config_defaults(partial_args.config_path)

    parser = argparse.ArgumentParser(description="Run batch inference and write predictions.jsonl")
    parser.set_defaults(**config_defaults)
    parser.add_argument("--config_path", type=Path, default=partial_args.config_path)
    parser.add_argument("--mode", type=str, default=None, choices=["base", "sft", "rag", "sft_rag"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--eval_file", type=Path, default=None)
    parser.add_argument("--output_file", type=Path, default=None)
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
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--use_rerank", action="store_true", help="Enable reranking after retrieval.")
    parser.add_argument("--rerank_model_name", type=str, default=None)
    parser.add_argument("--rerank_top_n", type=int, default=0)
    parser.add_argument("--rerank_keep_n", type=int, default=0)
    parser.add_argument("--rerank_score_threshold", type=float, default=None)
    parser.add_argument("--allow_ignore_context", action="store_true")
    parser.add_argument("--max_context_chunks", type=int, default=0)
    parser.add_argument("--log_rag_debug", action="store_true")
    parser.add_argument("--reranker_type", type=str, default="cross_encoder")
    parser.add_argument("--rerank_max_length", type=int, default=512)
    parser.add_argument("--rerank_batch_size", type=int, default=16)
    parser.add_argument("--output_token_logprobs", action="store_true")
    parser.add_argument("--logprob_last_k", type=int, default=5)
    return parser.parse_args()


def load_config_defaults(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"Inference config not found: {config_path}")

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pyyaml is required to load --config_path YAML files.") from exc

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict in inference config YAML: {config_path}")

    aliases = {
        "model_name_or_path": "model_path",
    }
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "uncertainty" and isinstance(value, dict):
            # Keep config-driven behavior simple: allow inference-time logprob toggles
            # under an uncertainty block.
            if "enabled" in value:
                normalized["output_token_logprobs"] = bool(value.get("enabled"))
            if "last_k" in value:
                normalized["logprob_last_k"] = int(value.get("last_k"))
            continue
        normalized[aliases.get(key, key)] = value
    return normalized


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


def validate_sample(sample: dict[str, Any], idx: int) -> None:
    if "question" not in sample:
        raise ValueError(f"Sample at index {idx} missing required field: question")


def main() -> None:
    args = parse_args()
    if not args.mode:
        raise ValueError("`mode` is required (CLI flag or config_path).")
    if not args.model_path:
        raise ValueError("`model_path` is required (CLI flag or config_path/model_name_or_path).")
    if args.eval_file is None:
        raise ValueError("`eval_file` is required (CLI flag or config_path).")
    if args.output_file is None:
        raise ValueError("`output_file` is required (CLI flag or config_path).")

    samples = load_eval_samples(args.eval_file)

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    for i, sample in enumerate(samples):
        validate_sample(sample, i)

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    predictions: list[dict[str, Any]] = []
    total = len(samples)

    for i, sample in enumerate(samples, start=1):
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
            use_rerank=args.use_rerank,
            rerank_model_name=args.rerank_model_name,
            rerank_top_n=args.rerank_top_n,
            rerank_keep_n=args.rerank_keep_n,
            rerank_score_threshold=args.rerank_score_threshold,
            allow_ignore_context=args.allow_ignore_context,
            max_context_chunks=args.max_context_chunks,
            log_rag_debug=args.log_rag_debug,
            reranker_type=args.reranker_type,
            rerank_max_length=args.rerank_max_length,
            rerank_batch_size=args.rerank_batch_size,
            output_token_logprobs=args.output_token_logprobs,
            logprob_last_k=args.logprob_last_k,
        )
        predictions.append(pred)

        if i % 50 == 0 or i == total:
            print(f"[batch_infer] Processed {i}/{total}")

    save_jsonl(predictions, args.output_file)
    print(f"[batch_infer] Wrote {len(predictions)} predictions to {args.output_file}")


if __name__ == "__main__":
    main()
