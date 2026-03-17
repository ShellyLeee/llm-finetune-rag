from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.rag.retrieve import retrieve


def load_model_and_tokenizer(model_path: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    adapter_cfg = Path(model_path) / "adapter_config.json"
    if adapter_cfg.exists():
        try:
            from peft import AutoPeftModelForCausalLM  # type: ignore

            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def extract_reference_fields(sample: dict[str, Any]) -> dict[str, Any]:
    answers = _as_list(sample.get("answers", []))
    answer_aliases = _as_list(sample.get("answer_aliases", []))
    normalized_answers = _as_list(sample.get("normalized_answers", []))
    normalized_aliases = _as_list(sample.get("normalized_aliases", []))

    gold_answer = answers[0] if answers else ""

    return {
        "gold_answer": gold_answer,
        "gold_answers": answers,
        "answer_aliases": answer_aliases,
        "normalized_answers": normalized_answers,
        "normalized_aliases": normalized_aliases,
    }


def build_prompt(question: str, mode: str, retrieved_docs: list[dict[str, Any]]) -> str:
    if mode in {"rag", "sft_rag"} and retrieved_docs:
        context = "\n\n".join(
            f"[{doc['chunk_id']}] {doc['text']}" for doc in retrieved_docs if doc.get("text")
        )
        return (
            "Answer the question using the provided context.\n"
            "Give only a short answer phrase.\n"
            "If the context is insufficient, output only: Insufficient context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    return (
        "Answer the question accurately.\n"
        "Give only a short answer phrase.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _normalize_retrieved_docs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items:
        normalized.append(
            {
                "doc_id": str(item.get("doc_id", item.get("chunk_id", ""))),
                "chunk_id": str(item.get("chunk_id", "")),
                "score": float(item.get("score", 0.0)),
                "rank": int(item.get("rank", 0)),
                "text": str(item.get("text", "")),
                "start": item.get("start"),
                "end": item.get("end"),
            }
        )
    return normalized


def _generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
) -> str:
    import torch

    do_sample = temperature > 0.0
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 简单清理，避免有些模型把 "Answer:" 又重复输出
    if text.lower().startswith("answer:"):
        text = text[len("answer:") :].strip()

    # 只保留第一行，减少 verbose 输出对 EM/F1 的伤害
    text = text.splitlines()[0].strip()

    return text


def generate_single_sample(
    sample: dict[str, Any],
    mode: str,
    model: Any,
    tokenizer: Any,
    top_k: int = 3,
    index_path: str = "data/corpus/indexes/wiki_demo.faiss",
    mapping_path: str = "data/corpus/chunks/wiki_demo_chunks.json",
    embedding_model_name: str | None = None,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
) -> dict[str, Any]:
    question = str(sample.get("question", ""))
    dataset = str(sample.get("dataset", ""))

    reference_fields = extract_reference_fields(sample)

    retrieved_docs: list[dict[str, Any]] = []
    if mode in {"rag", "sft_rag"}:
        retrieved_docs = _normalize_retrieved_docs(
            retrieve(
                query=question,
                top_k=top_k,
                index_path=index_path,
                mapping_path=mapping_path,
                embedding_model_name=embedding_model_name,
            )
        )

    prompt = build_prompt(question=question, mode=mode, retrieved_docs=retrieved_docs)
    prediction = _generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    return {
        "id": str(sample.get("id", "")),
        "dataset": dataset,
        "question": question,
        **reference_fields,
        "mode": mode,
        "prediction": prediction,
        "retrieved_docs": retrieved_docs if mode in {"rag", "sft_rag"} else [],
    }


def save_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")