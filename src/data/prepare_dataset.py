from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SFT dataset into JSONL for LLaMA-Factory.")
    parser.add_argument("--input", required=True, help="Path to input jsonl/csv file.")
    parser.add_argument("--output", required=True, help="Path to output jsonl file.")
    parser.add_argument("--dataset_name", default="demo_sft", help="Logical dataset name.")
    parser.add_argument("--lang_field", default=None, help="Optional language field name.")
    parser.add_argument("--max_len", type=int, default=None, help="Optional max char length for instruction/output.")
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported input format: {path.suffix}. Use .jsonl or .csv")


def to_messages(record: dict[str, Any], lang_field: str | None, max_len: int | None) -> dict[str, Any] | None:
    instruction = str(record.get("instruction", "")).strip()
    output = str(record.get("output", "")).strip()
    if not instruction or not output:
        return None
    if max_len is not None:
        instruction = instruction[:max_len]
        output = output[:max_len]
    sample = {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
    }
    if lang_field and record.get(lang_field):
        sample["lang"] = record[lang_field]
    return sample


def compute_stats(samples: list[dict[str, Any]], total_input_rows: int) -> dict[str, Any]:
    prompt_lengths = [len(item["messages"][0]["content"]) for item in samples]
    response_lengths = [len(item["messages"][1]["content"]) for item in samples]
    unique_pairs = {
        (item["messages"][0]["content"], item["messages"][1]["content"]) for item in samples
    }
    kept = len(samples)
    dedup_ratio = 0.0 if kept == 0 else round(len(unique_pairs) / kept, 4)
    return {
        "input_rows": total_input_rows,
        "kept_rows": kept,
        "dropped_rows": total_input_rows - kept,
        "dedup_ratio_placeholder": dedup_ratio,
        "prompt_length_chars": summarize_lengths(prompt_lengths),
        "response_length_chars": summarize_lengths(response_lengths),
        "top_languages": Counter(item.get("lang", "unknown") for item in samples).most_common(5),
    }


def summarize_lengths(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {"min": 0, "p50": 0, "p95": 0, "max": 0}
    ordered = sorted(lengths)
    return {
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max": ordered[-1],
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "processed/train" in output_path.as_posix():
        stats_path = output_path.parents[1] / "stats" / "data_stats.json"
    else:
        stats_path = output_path.parent / "data_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        raw_records = read_records(input_path)
    except ValueError as exc:
        print(f"[prepare_dataset] {exc}")
        print("[prepare_dataset] Next step: provide a .jsonl or .csv file with instruction/output columns.")
        return

    if not raw_records:
        stats = {
            "dataset_name": args.dataset_name,
            "input_rows": 0,
            "kept_rows": 0,
            "next_step": "Place your raw data under data/raw and rerun this script with --input.",
        }
        output_path.write_text("", encoding="utf-8")
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[prepare_dataset] No input data found at {input_path}.")
        print("[prepare_dataset] Wrote empty placeholder outputs.")
        print("[prepare_dataset] Next step: prepare a jsonl/csv file with instruction/output fields.")
        return

    samples = []
    for record in raw_records:
        sample = to_messages(record, args.lang_field, args.max_len)
        if sample is not None:
            samples.append(sample)

    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    stats = {"dataset_name": args.dataset_name, **compute_stats(samples, len(raw_records))}
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[prepare_dataset] Wrote {len(samples)} samples to {output_path}.")
    print(f"[prepare_dataset] Stats saved to {stats_path}.")
    if len(samples) == 0:
        print("[prepare_dataset] Next step: ensure each row has non-empty instruction/output fields.")
    else:
        print("[prepare_dataset] Next step: update data/dataset_info.json if you rename the dataset or split files.")


if __name__ == "__main__":
    main()
