# 本文件仅作RAG流程测试

from __future__ import annotations

import argparse
import json
from typing import Any

from src.rag.retrieve import retrieve


def build_rag_prompt(query: str, contexts: list[dict[str, Any]]) -> str:
    context_block = "\n\n".join(f"[{item['chunk_id']}] {item['text']}" for item in contexts)
    return (
        "你是一个RAG问答助手。请严格基于给定上下文回答问题，不要引入上下文之外的事实。\n"
        "回答要求：\n"
        "1) 必须在每个关键结论后标注引用的 chunk_id，格式如 [chunk-000123]\n"
        "2) 如果上下文不足，请明确说“根据当前检索片段无法确定”\n\n"
        f"【Context】\n{context_block}\n\n"
        f"【Question】\n{query}\n\n"
        "【Answer】"
    )


def dummy_generate(prompt: str, contexts: list[dict[str, Any]]) -> str:
    if not contexts:
        return "根据当前检索片段无法确定。"
    lead = contexts[0]["chunk_id"]
    return f"这是一个冒烟测试占位回答，信息来自检索片段 [{lead}]。"


def generate_answer(
    query: str,
    top_k: int = 3,
    index_path: str = "data/corpus/indexes/wiki_demo.faiss",
    mapping_path: str = "data/corpus/chunks/wiki_demo_chunks.json",
    embedding_model_name: str | None = None,
    contexts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    retrieved = contexts or retrieve(
        query=query,
        top_k=top_k,
        index_path=index_path,
        mapping_path=mapping_path,
        embedding_model_name=embedding_model_name,
    )
    prompt = build_rag_prompt(query=query, contexts=retrieved)
    answer = dummy_generate(prompt, retrieved)
    return {
        "query": query,
        "prompt": prompt,
        "answer": answer,
        "retrieved_chunks": retrieved,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate answer from RAG contexts (smoke test).")
    parser.add_argument("--query", type=str, required=True, help="User query text.")
    parser.add_argument("--top-k", type=int, default=3, help="Top K retrieval count.")
    parser.add_argument(
        "--index-path",
        type=str,
        default="data/corpus/indexes/wiki_demo.faiss",
        help="FAISS index path.",
    )
    parser.add_argument(
        "--mapping-path",
        type=str,
        default="data/corpus/chunks/wiki_demo_chunks.json",
        help="Chunk mapping path.",
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default=None,
        help="Optional embedding model override.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = generate_answer(
        query=args.query,
        top_k=args.top_k,
        index_path=args.index_path,
        mapping_path=args.mapping_path,
        embedding_model_name=args.embedding_model_name,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
