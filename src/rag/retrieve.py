from __future__ import annotations

from typing import Any


def retrieve(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": f"dummy-{idx}",
            "text": f"Placeholder retrieved chunk {idx} for query: {query}",
            "score": round(1.0 / (idx + 1), 4),
        }
        for idx in range(top_k)
    ]


if __name__ == "__main__":
    for item in retrieve("what is sft?"):
        print(item)

