import argparse
import json
import os
import random
from datasets import load_dataset

SYSTEM_PROMPT = "You are a factual question answering assistant. Answer accurately and concisely."


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def filter_valid(example):
    if not example.get("question"):
        return False
    answer = example.get("answer", {})
    if not isinstance(answer, dict):
        return False
    value = answer.get("value", "")
    return isinstance(value, str) and len(value.strip()) > 0


def make_sft_example(example):
    question = example["question"].strip()
    answer = example["answer"]["value"].strip()

    return {
        "instruction": "Answer the following question accurately and concisely.",
        "input": f"Question: {question}",
        "output": answer,
        "system": SYSTEM_PROMPT,
    }


def make_eval_example(example):
    answer_obj = example["answer"]
    return {
        "id": example["question_id"],
        "dataset": "triviaqa_rc_nocontext",
        "question": example["question"].strip(),
        "answers": [answer_obj["value"].strip()],
        "answer_aliases": answer_obj.get("aliases", []),
        "normalized_answers": [answer_obj.get("normalized_value", "").strip()],
        "normalized_aliases": answer_obj.get("normalized_aliases", []),
        "question_source": example.get("question_source", ""),
    }


def maybe_subsample(ds, max_samples, seed):
    if max_samples is None or max_samples >= len(ds):
        return ds
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices = indices[:max_samples]
    return ds.select(indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/sharedata/ssd_large/common/datasets/trivia_qa/rc.nocontext",
    )
    parser.add_argument("--output_root", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    train_dir = os.path.join(args.output_root, "train")
    eval_dir = os.path.join(args.output_root, "eval")
    stats_dir = os.path.join(args.output_root, "stats")

    ensure_dir(train_dir)
    ensure_dir(eval_dir)
    ensure_dir(stats_dir)

    ds = load_dataset(
        "parquet",
        data_files={
            "train": os.path.join(args.data_dir, "train-*.parquet"),
            "validation": os.path.join(args.data_dir, "validation-*.parquet"),
            "test": os.path.join(args.data_dir, "test-*.parquet"),
        },
    )

    ds = ds.filter(filter_valid, num_proc=8)

    total_train_available = len(ds["train"])
    total_val_available = len(ds["validation"])
    total_test_available = len(ds["test"])

    train_ds = maybe_subsample(ds["train"], args.max_train_samples, args.seed)
    val_ds = maybe_subsample(ds["validation"], args.max_val_samples, args.seed)
    test_ds = maybe_subsample(ds["test"], args.max_test_samples, args.seed)

    sft_train = [make_sft_example(x) for x in train_ds]
    sft_val = [make_sft_example(x) for x in val_ds]

    eval_val = [make_eval_example(x) for x in val_ds]
    eval_test = [make_eval_example(x) for x in test_ds]

    train_output = os.path.join(train_dir, "triviaqa_train.json")
    val_output = os.path.join(train_dir, "triviaqa_val.json")
    val_eval_output = os.path.join(eval_dir, "triviaqa_val_eval.jsonl")
    test_eval_output = os.path.join(eval_dir, "triviaqa_test_eval.jsonl")
    stats_output = os.path.join(stats_dir, "triviaqa_stats.json")

    save_json(sft_train, train_output)
    save_json(sft_val, val_output)
    save_jsonl(eval_val, val_eval_output)
    save_jsonl(eval_test, test_eval_output)

    stats = {
        "source_dir": args.data_dir,
        "dataset_name": "triviaqa_rc_nocontext",
        "seed": args.seed,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_test_samples": args.max_test_samples,
        "total_train_available": total_train_available,
        "total_val_available": total_val_available,
        "total_test_available": total_test_available,
        "train_samples": len(sft_train),
        "val_samples": len(sft_val),
        "test_samples": len(eval_test),
        "train_output": train_output,
        "val_output": val_output,
        "val_eval_output": val_eval_output,
        "test_eval_output": test_eval_output
    }
    save_json(stats, stats_output)

    print("Done.")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()