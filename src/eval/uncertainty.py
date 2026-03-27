from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UncertaintyConfig:
    enabled: bool = False
    score_type: str = "avg_logprob"
    last_k: int = 5
    score_weights: dict[str, float] | None = None


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def compute_logprob_features(
    token_logprobs: list[float],
    last_k: int = 5,
) -> dict[str, float | None]:
    if not token_logprobs:
        return {
            "avg_logprob": None,
            "min_logprob": None,
            "last_k_avg_logprob": None,
            "length_normalized_nll": None,
        }

    safe_last_k = max(1, int(last_k))
    tail = token_logprobs[-safe_last_k:]
    avg_logprob = safe_mean(token_logprobs)
    min_logprob = float(min(token_logprobs))
    last_k_avg_logprob = safe_mean(tail)
    length_normalized_nll = float(-sum(token_logprobs) / len(token_logprobs))

    return {
        "avg_logprob": avg_logprob,
        "min_logprob": min_logprob,
        "last_k_avg_logprob": last_k_avg_logprob,
        "length_normalized_nll": length_normalized_nll,
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_confidence_score(
    row: dict[str, Any],
    score_type: str,
    last_k: int = 5,
    score_weights: dict[str, float] | None = None,
) -> float | None:
    avg_logprob = _float_or_none(row.get("avg_logprob"))
    min_logprob = _float_or_none(row.get("min_logprob"))
    last_k_avg_logprob = _float_or_none(row.get("last_k_avg_logprob"))
    length_normalized_nll = _float_or_none(row.get("length_normalized_nll"))

    token_logprobs = row.get("token_logprobs")
    if isinstance(token_logprobs, list) and token_logprobs and (
        avg_logprob is None or min_logprob is None or last_k_avg_logprob is None or length_normalized_nll is None
    ):
        computed = compute_logprob_features([float(x) for x in token_logprobs], last_k=last_k)
        avg_logprob = _float_or_none(computed["avg_logprob"])
        min_logprob = _float_or_none(computed["min_logprob"])
        last_k_avg_logprob = _float_or_none(computed["last_k_avg_logprob"])
        length_normalized_nll = _float_or_none(computed["length_normalized_nll"])

    if score_type == "avg_logprob":
        return avg_logprob
    if score_type == "min_logprob":
        return min_logprob
    if score_type == "last_k_avg_logprob":
        return last_k_avg_logprob
    if score_type == "length_normalized_nll":
        if length_normalized_nll is None:
            return None
        # Convert uncertainty to confidence, so higher is always better.
        return float(-length_normalized_nll)
    if score_type == "weighted":
        weights = score_weights or {}
        terms = {
            "avg_logprob": avg_logprob,
            "min_logprob": min_logprob,
            "last_k_avg_logprob": last_k_avg_logprob,
            "length_normalized_nll": None if length_normalized_nll is None else -length_normalized_nll,
        }
        total_weight = 0.0
        total_score = 0.0
        for key, weight in weights.items():
            value = terms.get(key)
            if value is None:
                continue
            w = float(weight)
            total_weight += w
            total_score += w * value
        if total_weight <= 0.0:
            return None
        return float(total_score / total_weight)
    raise ValueError(f"Unsupported uncertainty score_type: {score_type}")
