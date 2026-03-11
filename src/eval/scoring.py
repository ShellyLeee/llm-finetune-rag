from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", "", text)
    return text

def exact_match(prediction: str, reference: str) -> float:
    return float(prediction.strip() == reference.strip())


def _lcs_length(a: str, b: str) -> int:
    if not a or not b:
        return 0
    n = len(b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, len(a) + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = curr[j - 1] if curr[j - 1] >= prev[j] else prev[j]
        prev, curr = curr, prev
    return prev[n]


def char_f1(prediction: str, reference: str) -> float:
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    ref_counts: dict[str, int] = {}
    for ch in ref:
        ref_counts[ch] = ref_counts.get(ch, 0) + 1

    overlap = 0
    for ch in pred:
        count = ref_counts.get(ch, 0)
        if count > 0:
            overlap += 1
            ref_counts[ch] = count - 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return float(2 * precision * recall / (precision + recall))


def rouge_l(prediction: str, reference: str) -> float:
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    lcs = _lcs_length(pred, ref)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
