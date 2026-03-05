from __future__ import annotations

import re
from typing import Any


def extract_chunk_ids(text: str) -> set[str]:
    pattern = r"\bchunk-[A-Za-z0-9_-]+\b"
    return set(re.findall(pattern, text))


"""
Minimal placeholder interface for hallucination-aware evaluation.

TODO:
- citation-based faithfulness checks✅ (Smoke test: chunk ID extraction + coverage)
- NLI / entailment-based support verification
- claim extraction + chunk grounding
"""

def compute_faithfulness(prediction: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    retrieved_ids = {
        str(item.get("chunk_id"))
        for item in retrieved_chunks
        if item.get("chunk_id")
    }
    cited_ids = extract_chunk_ids(prediction)
    covered_ids = cited_ids.intersection(retrieved_ids)

    if retrieved_ids:
        coverage = len(covered_ids) / len(retrieved_ids)
    else:
        coverage = 0.0

    return {
        "faithfulness_score": float(coverage),
        "citation_coverage": float(coverage),
        "retrieved_chunk_count": len(retrieved_ids),
        "cited_chunk_count": len(cited_ids),
        "covered_chunk_count": len(covered_ids),
        "retrieved_chunk_ids": sorted(retrieved_ids),
        "cited_chunk_ids": sorted(cited_ids),
        "covered_chunk_ids": sorted(covered_ids),
        "unsupported_claims": [],
    }
