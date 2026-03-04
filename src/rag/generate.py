from __future__ import annotations

from src.rag.retrieve import retrieve


def generate_answer(query: str) -> dict:
    chunks = retrieve(query)
    answer = "This is a placeholder answer generated from retrieved chunks."
    return {"query": query, "answer": answer, "retrieved_chunks": chunks}


if __name__ == "__main__":
    print(generate_answer("Explain LoRA."))

