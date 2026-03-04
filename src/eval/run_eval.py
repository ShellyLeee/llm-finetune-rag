from __future__ import annotations

import json
from pathlib import Path

from src.eval.hallucination import compute_faithfulness
from src.eval.scoring import exact_match
from src.rag.retrieve import retrieve


def ensure_dummy_eval_data(eval_dir: Path) -> Path:
    eval_dir.mkdir(parents=True, exist_ok=True)
    sample_path = eval_dir / "dummy_eval.jsonl"
    if sample_path.exists():
        return sample_path

    samples = [
        {
            "id": "dummy-1",
            "question": "What is supervised fine-tuning?",
            "reference": "Supervised fine-tuning adapts a pretrained model using labeled instruction-response data."
        },
        {
            "id": "dummy-2",
            "question": "What does RAG combine?",
            "reference": "RAG combines retrieval with generation."
        },
    ]
    with sample_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[run_eval] No eval data found. Created dummy samples at {sample_path}.")
    return sample_path


def load_eval_samples(eval_dir: Path) -> list[dict]:
    jsonl_files = sorted(eval_dir.glob("*.jsonl"))
    if not jsonl_files:
        jsonl_files = [ensure_dummy_eval_data(eval_dir)]

    samples = []
    for path in jsonl_files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    return samples


def placeholder_predict(question: str, chunks: list[dict]) -> str:
    if chunks:
        return f"Placeholder answer for: {question} | grounded on: {chunks[0]['text']}"
    return f"Placeholder answer for: {question}"


def main() -> None:
    root = Path.cwd()
    eval_dir = root / "data" / "eval"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    samples = load_eval_samples(eval_dir)
    results_path = reports_dir / "results.jsonl"
    summary_path = reports_dir / "summary.json"

    results = []
    for sample in samples:
        question = sample.get("question", "")
        reference = sample.get("reference", "")
        chunks = retrieve(question, top_k=3)
        prediction = placeholder_predict(question, chunks)
        faithfulness = compute_faithfulness(prediction, chunks)
        result = {
            "id": sample.get("id"),
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "retrieved_chunks": chunks,
            "metrics": {
                "exact_match": exact_match(prediction, reference),
                **faithfulness,
            },
        }
        results.append(result)

    with results_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = {
        "num_samples": len(results),
        "avg_exact_match": round(sum(r["metrics"]["exact_match"] for r in results) / max(len(results), 1), 4),
        "avg_faithfulness": round(
            sum(r["metrics"]["faithfulness_score"] for r in results) / max(len(results), 1), 4
        ),
        "next_step": "Replace placeholder_predict with real model inference and swap dummy retrieval with a real index.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[run_eval] Wrote per-sample results to {results_path}.")
    print(f"[run_eval] Wrote summary to {summary_path}.")


if __name__ == "__main__":
    main()

