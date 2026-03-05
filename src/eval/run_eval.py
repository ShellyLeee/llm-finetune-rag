from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.eval.hallucination import compute_faithfulness
from src.eval.scoring import exact_match
from src.rag.generate import generate_answer


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


def load_eval_samples(eval_dir: Path) -> list[dict[str, Any]]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG-aware evaluation.")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval"), help="Directory with eval jsonl files.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"), help="Directory to write reports.")
    parser.add_argument("--top-k", type=int, default=3, help="Top K retrieval for each sample.")
    parser.add_argument("--index-path", type=str, default="data/rag/wiki_demo.faiss", help="FAISS index path.")
    parser.add_argument(
        "--mapping-path",
        type=str,
        default="data/rag/wiki_demo_chunks.json",
        help="Chunk mapping path.",
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default=None,
        help="Optional embedding model override.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Limit eval samples; 0 means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = args.eval_dir
    reports_dir = args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    samples = load_eval_samples(eval_dir)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    results_path = reports_dir / "results.jsonl"
    summary_path = reports_dir / "summary.json"

    results = []
    for sample in samples:
        question = sample.get("question", "")
        reference = sample.get("reference", "")
        rag_output = generate_answer(
            query=question,
            top_k=args.top_k,
            index_path=args.index_path,
            mapping_path=args.mapping_path,
            embedding_model_name=args.embedding_model_name,
        )
        prediction = rag_output["answer"]
        chunks = rag_output["retrieved_chunks"]
        faithfulness = compute_faithfulness(prediction, chunks)
        result = {
            "id": sample.get("id"),
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "prompt": rag_output["prompt"],
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
        "top_k": args.top_k,
        "index_path": args.index_path,
        "mapping_path": args.mapping_path,
        "next_step": "Replace dummy_generate with real model inference backend while keeping citation format.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[run_eval] Wrote per-sample results to {results_path}.")
    print(f"[run_eval] Wrote summary to {summary_path}.")


if __name__ == "__main__":
    main()
