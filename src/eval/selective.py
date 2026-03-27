from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SelectiveConfig:
    enabled: bool = False
    mode: str = "fixed"  # fixed | target_coverage
    threshold: float = -2.0
    target_coverage: float = 1.0
    refusal_text: str = "I'm not confident enough to answer this reliably."
    ood_delta: float = 0.0


def select_threshold_by_coverage(confidence_scores: list[float], target_coverage: float) -> float:
    if not confidence_scores:
        return math.inf

    clamped_cov = max(0.0, min(1.0, float(target_coverage)))
    if clamped_cov <= 0.0:
        return math.inf
    if clamped_cov >= 1.0:
        return float(min(confidence_scores))

    sorted_scores = sorted(confidence_scores, reverse=True)
    keep = max(1, int(math.ceil(clamped_cov * len(sorted_scores))))
    return float(sorted_scores[keep - 1])


def resolve_base_threshold(
    confidence_scores: list[float],
    mode: str,
    fixed_threshold: float,
    target_coverage: float,
) -> float:
    if mode == "fixed":
        return float(fixed_threshold)
    if mode == "target_coverage":
        return select_threshold_by_coverage(confidence_scores, target_coverage=target_coverage)
    raise ValueError(f"Unsupported selective mode: {mode}")


def threshold_for_split(base_threshold: float, split_type: str, ood_delta: float) -> float:
    if split_type.lower() == "ood":
        return float(base_threshold + float(ood_delta))
    return float(base_threshold)


def apply_abstention(
    confidence_score: float | None,
    threshold: float,
) -> tuple[bool, str]:
    if confidence_score is None:
        return True, "missing_confidence"
    if confidence_score < threshold:
        return True, "below_threshold"
    return False, ""


def split_confidences(rows: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        score = row.get("confidence_score")
        if score is None:
            continue
        values.append(float(score))
    return values
