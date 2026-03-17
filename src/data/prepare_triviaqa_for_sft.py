import argparse
import json
import os
from datasets import load_dataset


SYSTEM_PROMPT = "You are a factual question answering assistant. Answer accurately and concisely."


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


def filter_valid(example):
    if not example.get("question"):
        return False
    answer = example.get("answer", {})
    if not isinstance(answer, dict):
        return False
    value = answer.get("value", "")
    return isinstance(value, str) and len(value.strip()) > 0


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/sharedata/ssd_large/common/datasets/trivia_qa/rc.nocontext",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_dataset(
        "parquet",
        data_files={
            "train": os.path.join(args.data_dir, "train-*.parquet"),
            "validation": os.path.join(args.data_dir, "validation-*.parquet"),
            "test": os.path.join(args.data_dir, "test-*.parquet"),
        },
    )

    # 过滤无效样本
    ds = ds.filter(filter_valid, num_proc=8)

    train_ds = ds["train"]
    val_ds = ds["validation"]
    test_ds = ds["test"]

    if args.max_train_samples is not None:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_val_samples is not None:
        val_ds = val_ds.select(range(min(args.max_val_samples, len(val_ds))))
    if args.max_test_samples is not None:
        test_ds = test_ds.select(range(min(args.max_test_samples, len(test_ds))))

    sft_train = [make_sft_example(x) for x in train_ds]
    sft_val = [make_sft_example(x) for x in val_ds]

    eval_val = [make_eval_example(x) for x in val_ds]
    eval_test = [make_eval_example(x) for x in test_ds]

    save_json(sft_train, os.path.join(args.output_dir, "triviaqa_rc_nocontext_train.json"))
    save_json(sft_val, os.path.join(args.output_dir, "triviaqa_rc_nocontext_val.json"))
    save_json(eval_val, os.path.join(args.output_dir, "triviaqa_rc_nocontext_val_eval.json"))
    save_json(eval_test, os.path.join(args.output_dir, "triviaqa_rc_nocontext_test_eval.json"))

    print("Done.")
    print(f"train: {len(sft_train)}")
    print(f"val:   {len(sft_val)}")
    print(f"test:  {len(eval_test)}")


if __name__ == "__main__":
    main()