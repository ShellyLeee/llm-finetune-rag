from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed train/eval split from alpaca-style JSON.")
    parser.add_argument(
        "--input_file",
        type=Path,
        default=Path("data/alpaca_zh_demo.json"),
        help="Path to source alpaca JSON file.",
    )
    parser.add_argument(
        "--train_output",
        type=Path,
        default=Path("data/processed/train/train.jsonl"),
        help="Path to train jsonl output.",
    )
    parser.add_argument(
        "--eval_output",
        type=Path,
        default=Path("data/processed/eval/eval.jsonl"),
        help="Path to eval jsonl output.",
    )
    parser.add_argument(
        "--stats_output",
        type=Path,
        default=Path("data/processed/stats/split_stats.json"),
        help="Path to split stats output.",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.1,
        help="Eval split ratio, default 0.1 (10%%).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_alpaca_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return [item for item in data if isinstance(item, dict)]


def build_question(instruction: str, input_text: str) -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        return f"{instruction}\n{input_text}" if instruction else input_text
    return instruction


def to_unified_schema(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, item in enumerate(records, start=1):
        instruction = str(item.get("instruction", "")).strip()
        input_text = str(item.get("input", "")).strip()
        output = str(item.get("output", "")).strip()
        question = build_question(instruction=instruction, input_text=input_text)

        out.append(
            {
                "id": f"alpaca_{i:06d}",
                "question": question,
                "gold_answer": output,
                "source": "alpaca_zh_demo",
            }
        )
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not (0.0 < args.eval_ratio < 1.0):
        raise ValueError("--eval_ratio must be in (0, 1).")

    raw = load_alpaca_json(args.input_file)
    records = to_unified_schema(raw)

    rng = random.Random(args.seed)
    rng.shuffle(records)

    total = len(records)
    eval_count = int(total * args.eval_ratio)
    train_count = total - eval_count
    train_rows = records[:train_count]
    eval_rows = records[train_count:]

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.eval_output, eval_rows)

    stats = {
        "source_file": str(args.input_file),
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "total_samples": total,
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "train_output": str(args.train_output),
        "eval_output": str(args.eval_output),
    }
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[build_eval_split] total={total} train={len(train_rows)} eval={len(eval_rows)} "
        f"seed={args.seed} eval_ratio={args.eval_ratio}"
    )
    print(f"[build_eval_split] train -> {args.train_output}")
    print(f"[build_eval_split] eval  -> {args.eval_output}")
    print(f"[build_eval_split] stats -> {args.stats_output}")


if __name__ == "__main__":
    main()
