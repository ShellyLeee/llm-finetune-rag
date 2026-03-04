from __future__ import annotations


def compute_faithfulness(prediction: str, retrieved_chunks: list[dict]) -> dict:
    """
    Minimal placeholder interface for hallucination-aware evaluation.

    TODO:
    - citation-based faithfulness checks
    - NLI / entailment-based support verification
    - claim extraction + chunk grounding
    """
    score = 0.0
    if prediction.strip() and retrieved_chunks:
        score = 0.5
    return {
        "faithfulness_score": float(score),
        "unsupported_claims": [],
    }

