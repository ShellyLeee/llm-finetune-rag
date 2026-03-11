from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.eval.hallucination import compute_faithfulness
from src.eval.scoring import char_f1, exact_match, rouge_l


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions.jsonl and write summary artifacts.")
    parser.add_argument("--predictions_file", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to predictions file parent directory.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    predictions = load_jsonl(args.predictions_file)
    output_dir = args.output_dir or args.predictions_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_match_scores: list[float] = []
    char_f1_scores: list[float] = []
    rouge_l_scores: list[float] = []
    faithfulness_scores: list[float] = []
    citation_coverages: list[float] = []
    per_sample: list[dict[str, Any]] = []

    for row in predictions:
        gold = str(row.get("gold_answer", ""))
        pred = str(row.get("prediction", ""))
        retrieved_docs = row.get("retrieved_docs", [])

        em = exact_match(pred, gold) if gold else 0.0
        cf1 = char_f1(pred, gold) if gold else 0.0
        rl = rouge_l(pred, gold) if gold else 0.0
        faithfulness = compute_faithfulness(prediction=pred, retrieved_chunks=retrieved_docs)

        exact_match_scores.append(em)
        char_f1_scores.append(cf1)
        rouge_l_scores.append(rl)
        faithfulness_scores.append(float(faithfulness["faithfulness_score"]))
        citation_coverages.append(float(faithfulness["citation_coverage"]))

        per_sample.append(
            {
                "id": row.get("id", ""),
                "mode": row.get("mode", ""),
                "exact_match": em,
                "char_f1": cf1,
                "rouge_l": rl,
                "faithfulness_score": faithfulness["faithfulness_score"],
                "citation_coverage": faithfulness["citation_coverage"],
            }
        )

    task_metrics = {
        "num_samples": len(predictions),
        "avg_exact_match": round(safe_mean(exact_match_scores), 4),
        "avg_char_f1": round(safe_mean(char_f1_scores), 4),
        "avg_rouge_l": round(safe_mean(rouge_l_scores), 4),
        "per_sample": per_sample,
    }
    hallucination_metrics = {
        "num_samples": len(predictions),
        "avg_faithfulness": round(safe_mean(faithfulness_scores), 4),
        "avg_citation_coverage": round(safe_mean(citation_coverages), 4),
    }
    summary = {
        "num_samples": len(predictions),
        "avg_exact_match": task_metrics["avg_exact_match"],
        "avg_char_f1": task_metrics["avg_char_f1"],
        "avg_rouge_l": task_metrics["avg_rouge_l"],
        "avg_faithfulness": hallucination_metrics["avg_faithfulness"],
        "avg_citation_coverage": hallucination_metrics["avg_citation_coverage"],
    }

    task_path = output_dir / "task_metrics.json"
    hallu_path = output_dir / "hallucination_metrics.json"
    summary_path = output_dir / "summary.json"

    write_json(task_path, task_metrics)
    write_json(hallu_path, hallucination_metrics)
    write_json(summary_path, summary)

    print(f"[run_eval] Wrote {task_path}")
    print(f"[run_eval] Wrote {hallu_path}")
    print(f"[run_eval] Wrote {summary_path}")


if __name__ == "__main__":
    main()