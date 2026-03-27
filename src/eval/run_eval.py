from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from src.eval.hallucination import compute_faithfulness
from src.eval.scoring import char_f1, exact_match, rouge_l
from src.eval.selective import (
    SelectiveConfig,
    apply_abstention,
    resolve_base_threshold,
    split_confidences,
    threshold_for_split,
)
from src.eval.selective_metrics import auprc_binary, auroc_binary, risk_coverage_curve
from src.eval.uncertainty import UncertaintyConfig, compute_confidence_score


def _str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def load_config_defaults(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"Eval config not found: {config_path}")

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pyyaml is required to load --config_path YAML files.") from exc

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict in eval config YAML: {config_path}")

    defaults: dict[str, Any] = {}
    if "predictions_file" in payload:
        defaults["predictions_file"] = Path(payload["predictions_file"])
    if "output_dir" in payload and payload["output_dir"] is not None:
        defaults["output_dir"] = Path(payload["output_dir"])
    if "split_type_field" in payload:
        defaults["split_type_field"] = str(payload["split_type_field"])
    if "ood_label" in payload:
        defaults["ood_label"] = str(payload["ood_label"])
    if "default_split_type" in payload:
        defaults["default_split_type"] = str(payload["default_split_type"])

    uncertainty = payload.get("uncertainty", {})
    if isinstance(uncertainty, dict):
        if "enabled" in uncertainty:
            defaults["uncertainty_enabled"] = _str_to_bool(uncertainty["enabled"])
        if "score_type" in uncertainty:
            defaults["uncertainty_score_type"] = str(uncertainty["score_type"])
        if "last_k" in uncertainty:
            defaults["uncertainty_last_k"] = int(uncertainty["last_k"])
        if "score_weights" in uncertainty and isinstance(uncertainty["score_weights"], dict):
            defaults["uncertainty_score_weights"] = json.dumps(uncertainty["score_weights"])

    selective = payload.get("selective", {})
    if isinstance(selective, dict):
        if "enabled" in selective:
            defaults["selective_enabled"] = _str_to_bool(selective["enabled"])
        if "mode" in selective:
            defaults["selective_mode"] = str(selective["mode"])
        if "threshold" in selective:
            defaults["selective_threshold"] = float(selective["threshold"])
        if "target_coverage" in selective:
            defaults["selective_target_coverage"] = float(selective["target_coverage"])
        if "refusal_text" in selective:
            defaults["selective_refusal_text"] = str(selective["refusal_text"])
        if "ood_delta" in selective:
            defaults["selective_ood_delta"] = float(selective["ood_delta"])

    return defaults


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config_path", type=Path, default=None)
    partial_args, _ = bootstrap.parse_known_args()
    defaults = load_config_defaults(partial_args.config_path)

    parser = argparse.ArgumentParser(description="Evaluate predictions.jsonl and write summary artifacts.")
    parser.set_defaults(**defaults)
    parser.add_argument("--config_path", type=Path, default=partial_args.config_path)
    parser.add_argument("--predictions_file", type=Path, default=None)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to predictions file parent directory.",
    )
    parser.add_argument("--split_type_field", type=str, default="split_type")
    parser.add_argument("--ood_label", type=str, default="ood")
    parser.add_argument("--default_split_type", type=str, default="id")

    parser.add_argument("--uncertainty_enabled", action="store_true")
    parser.add_argument(
        "--uncertainty_score_type",
        type=str,
        default="avg_logprob",
        choices=["avg_logprob", "min_logprob", "last_k_avg_logprob", "length_normalized_nll", "weighted"],
    )
    parser.add_argument("--uncertainty_last_k", type=int, default=5)
    parser.add_argument(
        "--uncertainty_score_weights",
        type=str,
        default="",
        help="JSON object for weighted score, e.g. '{\"avg_logprob\":0.7,\"min_logprob\":0.3}'.",
    )

    parser.add_argument("--selective_enabled", action="store_true")
    parser.add_argument("--selective_mode", type=str, default="fixed", choices=["fixed", "target_coverage"])
    parser.add_argument("--selective_threshold", type=float, default=-2.0)
    parser.add_argument("--selective_target_coverage", type=float, default=1.0)
    parser.add_argument(
        "--selective_refusal_text",
        type=str,
        default="I'm not confident enough to answer this reliably.",
    )
    parser.add_argument("--selective_ood_delta", type=float, default=0.0)
    parser.add_argument("--disable_risk_coverage_curve", action="store_true")
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


def save_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_split(value: Any, default_split: str) -> str:
    text = str(value or default_split).strip().lower()
    return text if text else default_split


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_score_weights(raw: str) -> dict[str, float] | None:
    if not raw:
        return None
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("uncertainty_score_weights must be a JSON object.")
    normalized: dict[str, float] = {}
    for key, value in payload.items():
        normalized[str(key)] = float(value)
    return normalized


def _build_summary_subset(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "coverage": 0.0,
            "abstention_rate": 0.0,
            "avg_exact_match_accepted": 0.0,
            "avg_char_f1_accepted": 0.0,
            "avg_rouge_l_accepted": 0.0,
            "accepted_error_rate": 0.0,
        }
    accepted = [r for r in rows if not bool(r.get("abstained", False))]
    avg_em = safe_mean([float(r.get("exact_match", 0.0)) for r in accepted])
    avg_cf1 = safe_mean([float(r.get("char_f1", 0.0)) for r in accepted])
    avg_rl = safe_mean([float(r.get("rouge_l", 0.0)) for r in accepted])
    accepted_error_rate = 1.0 - avg_em if accepted else 0.0
    return {
        "coverage": round(len(accepted) / len(rows), 4),
        "abstention_rate": round(1.0 - (len(accepted) / len(rows)), 4),
        "avg_exact_match_accepted": round(avg_em, 4),
        "avg_char_f1_accepted": round(avg_cf1, 4),
        "avg_rouge_l_accepted": round(avg_rl, 4),
        "accepted_error_rate": round(accepted_error_rate, 4),
    }


def main() -> None:
    args = parse_args()
    if args.predictions_file is None:
        raise ValueError("`predictions_file` is required (CLI flag or config_path).")
    predictions = load_jsonl(args.predictions_file)
    output_dir = args.output_dir or args.predictions_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    score_weights = _parse_score_weights(args.uncertainty_score_weights)
    uncertainty_cfg = UncertaintyConfig(
        enabled=bool(args.uncertainty_enabled),
        score_type=args.uncertainty_score_type,
        last_k=max(1, int(args.uncertainty_last_k)),
        score_weights=score_weights,
    )
    selective_cfg = SelectiveConfig(
        enabled=bool(args.selective_enabled),
        mode=args.selective_mode,
        threshold=float(args.selective_threshold),
        target_coverage=float(args.selective_target_coverage),
        refusal_text=str(args.selective_refusal_text),
        ood_delta=float(args.selective_ood_delta),
    )

    enriched_rows: list[dict[str, Any]] = []
    for row in predictions:
        split_type = _normalize_split(row.get(args.split_type_field), args.default_split_type)
        confidence_score: float | None = None
        if uncertainty_cfg.enabled:
            confidence_score = compute_confidence_score(
                row=row,
                score_type=uncertainty_cfg.score_type,
                last_k=uncertainty_cfg.last_k,
                score_weights=uncertainty_cfg.score_weights,
            )

        enriched = dict(row)
        enriched["split_type"] = split_type
        enriched["confidence_score"] = confidence_score
        enriched_rows.append(enriched)

    base_threshold = selective_cfg.threshold
    if selective_cfg.enabled:
        base_threshold = resolve_base_threshold(
            confidence_scores=split_confidences(enriched_rows),
            mode=selective_cfg.mode,
            fixed_threshold=selective_cfg.threshold,
            target_coverage=selective_cfg.target_coverage,
        )

    exact_match_scores: list[float] = []
    char_f1_scores: list[float] = []
    rouge_l_scores: list[float] = []
    faithfulness_scores: list[float] = []
    citation_coverages: list[float] = []
    accepted_em: list[float] = []
    accepted_cf1: list[float] = []
    accepted_rl: list[float] = []
    accepted_hallu: list[float] = []
    all_hallu: list[float] = []

    id_rows: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []

    error_labels: list[int] = []
    error_scores: list[float] = []
    confidences_for_curve: list[float] = []
    incorrect_for_curve: list[int] = []

    for row in enriched_rows:
        gold = str(row.get("gold_answer", ""))
        pred_raw = str(row.get("prediction_raw", row.get("prediction", "")))
        retrieved_docs = row.get("retrieved_docs", [])
        confidence_score = _to_float_or_none(row.get("confidence_score"))
        split_type = _normalize_split(row.get("split_type"), args.default_split_type)

        threshold_used = None
        abstained = False
        abstain_reason = ""
        prediction_final = pred_raw
        if selective_cfg.enabled:
            threshold_used = threshold_for_split(
                base_threshold=base_threshold,
                split_type=split_type,
                ood_delta=selective_cfg.ood_delta,
            )
            abstained, abstain_reason = apply_abstention(confidence_score=confidence_score, threshold=threshold_used)
            if abstained:
                prediction_final = selective_cfg.refusal_text
        else:
            abstain_reason = "selective_disabled"

        em = exact_match(pred_raw, gold) if gold else 0.0
        cf1 = char_f1(pred_raw, gold) if gold else 0.0
        rl = rouge_l(pred_raw, gold) if gold else 0.0
        is_correct = int(em >= 1.0)
        is_hallucinated = int(1 - is_correct)

        faithfulness = compute_faithfulness(prediction=pred_raw, retrieved_chunks=retrieved_docs)

        exact_match_scores.append(em)
        char_f1_scores.append(cf1)
        rouge_l_scores.append(rl)
        faithfulness_scores.append(float(faithfulness["faithfulness_score"]))
        citation_coverages.append(float(faithfulness["citation_coverage"]))
        all_hallu.append(float(is_hallucinated))
        if not abstained:
            accepted_em.append(em)
            accepted_cf1.append(cf1)
            accepted_rl.append(rl)
            accepted_hallu.append(float(is_hallucinated))

        incorrect = int(1 - is_correct)
        if confidence_score is not None and not math.isinf(confidence_score) and not math.isnan(confidence_score):
            error_labels.append(incorrect)
            error_scores.append(-confidence_score)
            confidences_for_curve.append(confidence_score)
            incorrect_for_curve.append(incorrect)

        row["prediction_raw"] = pred_raw
        row["prediction_final"] = prediction_final
        row["prediction"] = prediction_final
        row["threshold_used"] = threshold_used
        row["abstained"] = abstained
        row["abstain_reason"] = abstain_reason
        row["exact_match"] = em
        row["char_f1"] = cf1
        row["rouge_l"] = rl
        row["is_correct"] = is_correct
        row["is_hallucinated"] = is_hallucinated
        row["faithfulness_score"] = faithfulness["faithfulness_score"]
        row["citation_coverage"] = faithfulness["citation_coverage"]

        if split_type == args.ood_label.lower():
            ood_rows.append(row)
        else:
            id_rows.append(row)

    num_samples = len(enriched_rows)
    accepted_rows = [r for r in enriched_rows if not bool(r.get("abstained", False))]
    coverage = (len(accepted_rows) / num_samples) if num_samples else 0.0
    abstention_rate = 1.0 - coverage

    avg_exact_match_accepted = safe_mean(accepted_em)
    avg_char_f1_accepted = safe_mean(accepted_cf1)
    avg_rouge_l_accepted = safe_mean(accepted_rl)
    selective_risk = 1.0 - avg_exact_match_accepted if accepted_rows else 0.0
    accepted_error_rate = 1.0 - avg_exact_match_accepted if accepted_rows else 0.0

    hallucination_rate_all = safe_mean(all_hallu)
    hallucination_rate_accepted = safe_mean(accepted_hallu)
    relative_hallu_reduction = 0.0
    if hallucination_rate_all > 0:
        relative_hallu_reduction = (hallucination_rate_all - hallucination_rate_accepted) / hallucination_rate_all

    subset_id = _build_summary_subset(id_rows)
    subset_ood = _build_summary_subset(ood_rows)

    task_metrics = {
        "num_samples": num_samples,
        "avg_exact_match": round(safe_mean(exact_match_scores), 4),
        "avg_char_f1": round(safe_mean(char_f1_scores), 4),
        "avg_rouge_l": round(safe_mean(rouge_l_scores), 4),
        "per_sample": [
            {
                "id": row.get("id", ""),
                "mode": row.get("mode", ""),
                "split_type": row.get("split_type", args.default_split_type),
                "exact_match": row.get("exact_match", 0.0),
                "char_f1": row.get("char_f1", 0.0),
                "rouge_l": row.get("rouge_l", 0.0),
                "confidence_score": row.get("confidence_score"),
                "abstained": row.get("abstained", False),
            }
            for row in enriched_rows
        ],
    }
    hallucination_metrics = {
        "num_samples": num_samples,
        "avg_faithfulness": round(safe_mean(faithfulness_scores), 4),
        "avg_citation_coverage": round(safe_mean(citation_coverages), 4),
        "hallucination_rate_all": round(hallucination_rate_all, 4),
        "hallucination_rate_accepted": round(hallucination_rate_accepted, 4),
        "relative_hallucination_reduction": round(relative_hallu_reduction, 4),
    }

    curve_payload = (
        {"coverage": [], "risk": [], "aurc": None}
        if args.disable_risk_coverage_curve
        else risk_coverage_curve(confidences_for_curve, incorrect_for_curve)
    )
    auroc = auroc_binary(error_labels, error_scores)
    auprc = auprc_binary(error_labels, error_scores)

    summary = {
        "num_samples": num_samples,
        "avg_exact_match": task_metrics["avg_exact_match"],
        "avg_char_f1": task_metrics["avg_char_f1"],
        "avg_rouge_l": task_metrics["avg_rouge_l"],
        "avg_faithfulness": hallucination_metrics["avg_faithfulness"],
        "avg_citation_coverage": hallucination_metrics["avg_citation_coverage"],
        "coverage": round(coverage, 4),
        "abstention_rate": round(abstention_rate, 4),
        "avg_exact_match_accepted": round(avg_exact_match_accepted, 4),
        "avg_char_f1_accepted": round(avg_char_f1_accepted, 4),
        "avg_rouge_l_accepted": round(avg_rouge_l_accepted, 4),
        "selective_risk": round(selective_risk, 4),
        "accepted_error_rate": round(accepted_error_rate, 4),
        "id_coverage": subset_id["coverage"],
        "ood_coverage": subset_ood["coverage"],
        "id_abstention_rate": subset_id["abstention_rate"],
        "ood_abstention_rate": subset_ood["abstention_rate"],
        "id_avg_exact_match_accepted": subset_id["avg_exact_match_accepted"],
        "ood_avg_exact_match_accepted": subset_ood["avg_exact_match_accepted"],
        "id_avg_char_f1_accepted": subset_id["avg_char_f1_accepted"],
        "ood_avg_char_f1_accepted": subset_ood["avg_char_f1_accepted"],
        "id_avg_rouge_l_accepted": subset_id["avg_rouge_l_accepted"],
        "ood_avg_rouge_l_accepted": subset_ood["avg_rouge_l_accepted"],
        "id_accepted_error_rate": subset_id["accepted_error_rate"],
        "ood_accepted_error_rate": subset_ood["accepted_error_rate"],
        "auroc_error_detection": None if auroc is None else round(auroc, 4),
        "auprc_error_detection": None if auprc is None else round(auprc, 4),
        "aurc": None if curve_payload["aurc"] is None else round(float(curve_payload["aurc"]), 4),
        "hallucination_rate_all": hallucination_metrics["hallucination_rate_all"],
        "hallucination_rate_accepted": hallucination_metrics["hallucination_rate_accepted"],
        "relative_hallucination_reduction": hallucination_metrics["relative_hallucination_reduction"],
    }

    selective_metrics = {
        "num_samples": num_samples,
        "uncertainty": {
            "enabled": uncertainty_cfg.enabled,
            "score_type": uncertainty_cfg.score_type,
            "last_k": uncertainty_cfg.last_k,
            "score_weights": uncertainty_cfg.score_weights,
        },
        "selective": {
            "enabled": selective_cfg.enabled,
            "mode": selective_cfg.mode,
            "base_threshold": base_threshold if selective_cfg.enabled else None,
            "fixed_threshold": selective_cfg.threshold,
            "target_coverage": selective_cfg.target_coverage,
            "ood_delta": selective_cfg.ood_delta,
        },
        "coverage": summary["coverage"],
        "abstention_rate": summary["abstention_rate"],
        "avg_exact_match_accepted": summary["avg_exact_match_accepted"],
        "avg_char_f1_accepted": summary["avg_char_f1_accepted"],
        "avg_rouge_l_accepted": summary["avg_rouge_l_accepted"],
        "selective_risk": summary["selective_risk"],
        "accepted_error_rate": summary["accepted_error_rate"],
        "id_coverage": summary["id_coverage"],
        "ood_coverage": summary["ood_coverage"],
        "id_abstention_rate": summary["id_abstention_rate"],
        "ood_abstention_rate": summary["ood_abstention_rate"],
        "id_avg_exact_match_accepted": summary["id_avg_exact_match_accepted"],
        "ood_avg_exact_match_accepted": summary["ood_avg_exact_match_accepted"],
        "id_avg_char_f1_accepted": summary["id_avg_char_f1_accepted"],
        "ood_avg_char_f1_accepted": summary["ood_avg_char_f1_accepted"],
        "id_avg_rouge_l_accepted": summary["id_avg_rouge_l_accepted"],
        "ood_avg_rouge_l_accepted": summary["ood_avg_rouge_l_accepted"],
        "id_accepted_error_rate": summary["id_accepted_error_rate"],
        "ood_accepted_error_rate": summary["ood_accepted_error_rate"],
        "auroc_error_detection": summary["auroc_error_detection"],
        "auprc_error_detection": summary["auprc_error_detection"],
        "aurc": summary["aurc"],
        "risk_coverage_curve": curve_payload,
    }

    task_path = output_dir / "task_metrics.json"
    hallu_path = output_dir / "hallucination_metrics.json"
    selective_path = output_dir / "selective_metrics.json"
    summary_path = output_dir / "summary.json"
    evaluated_predictions_path = output_dir / "predictions_evaluated.jsonl"

    write_json(task_path, task_metrics)
    write_json(hallu_path, hallucination_metrics)
    write_json(selective_path, selective_metrics)
    write_json(summary_path, summary)
    save_jsonl(enriched_rows, evaluated_predictions_path)

    print(f"[run_eval] Wrote {task_path}")
    print(f"[run_eval] Wrote {hallu_path}")
    print(f"[run_eval] Wrote {selective_path}")
    print(f"[run_eval] Wrote {summary_path}")
    print(f"[run_eval] Wrote {evaluated_predictions_path}")


if __name__ == "__main__":
    main()
