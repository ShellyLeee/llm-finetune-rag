from __future__ import annotations

from typing import Any


def risk_coverage_curve(
    confidences: list[float],
    incorrect_labels: list[int],
) -> dict[str, Any]:
    if not confidences or len(confidences) != len(incorrect_labels):
        return {"coverage": [], "risk": [], "aurc": None}

    ranked = sorted(zip(confidences, incorrect_labels), key=lambda x: x[0], reverse=True)
    total = len(ranked)
    cum_errors = 0
    coverage: list[float] = []
    risk: list[float] = []

    for i, (_, err) in enumerate(ranked, start=1):
        cum_errors += int(err)
        coverage.append(i / total)
        risk.append(cum_errors / i)

    aurc = sum(risk) / total
    return {"coverage": coverage, "risk": risk, "aurc": float(aurc)}


def auroc_binary(labels: list[int], scores: list[float]) -> float | None:
    if not labels or len(labels) != len(scores):
        return None

    pos = sum(1 for x in labels if x == 1)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None

    ranked = sorted(zip(scores, labels), key=lambda x: x[0])
    n = len(ranked)
    rank_sum_pos = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        count_pos = sum(1 for _, label in ranked[i : j + 1] if label == 1)
        rank_sum_pos += avg_rank * count_pos
        i = j + 1

    auc = (rank_sum_pos - (pos * (pos + 1) / 2.0)) / (pos * neg)
    return float(auc)


def auprc_binary(labels: list[int], scores: list[float]) -> float | None:
    if not labels or len(labels) != len(scores):
        return None

    total_pos = sum(1 for x in labels if x == 1)
    if total_pos == 0:
        return None

    ranked = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0

    for _, label in ranked:
        if label == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / total_pos
        precision = tp / (tp + fp)
        area += (recall - prev_recall) * precision
        prev_recall = recall

    return float(area)
