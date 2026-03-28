from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.eval.selective import apply_abstention, threshold_for_split
from src.eval.uncertainty import compute_logprob_features
from src.eval.uncertainty import compute_confidence_score
from src.rag.rerank import build_reranker, filter_reranked_docs
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


def build_prompt(
    question: str,
    mode: str,
    retrieved_docs: list[dict[str, Any]],
    allow_ignore_context: bool = False,
) -> str:
    if mode in {"rag", "sft_rag"} and retrieved_docs:
        context = "\n\n".join(
            f"[{doc['chunk_id']}] {doc['text']}" for doc in retrieved_docs if doc.get("text")
        )
        if allow_ignore_context:
            return (
                "Answer the question concisely.\n"
                "Use retrieved context only if it is relevant.\n"
                "If context is irrelevant or insufficient, ignore it and use your own knowledge.\n"
                "Do not use unrelated retrieved text as evidence.\n"
                "Give only a short answer phrase.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n"
                "Answer:"
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


def _apply_rerank(
    query: str,
    retrieved_docs: list[dict[str, Any]],
    use_rerank: bool,
    rerank_model_name: str | None,
    rerank_top_n: int,
    rerank_keep_n: int,
    rerank_score_threshold: float | None,
    reranker_type: str = "cross_encoder",
    rerank_max_length: int = 512,
    rerank_batch_size: int = 16,
    max_context_chunks: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Optionally rerank/filter docs and return selected docs with debug info."""
    candidate_n = max(0, int(rerank_top_n))
    keep_n = max(0, int(rerank_keep_n))
    threshold = float(rerank_score_threshold) if rerank_score_threshold is not None else None

    debug: dict[str, Any] = {
        "retrieved_chunk_count": len(retrieved_docs),
        "rerank_enabled": bool(use_rerank),
        "rerank_candidate_count": 0,
        "reranked_chunk_count": len(retrieved_docs),
        "top_rerank_scores": [],
        "fallback_no_context": False,
    }

    if not retrieved_docs:
        debug["fallback_no_context"] = True
        return [], debug

    selected_docs = list(retrieved_docs)
    if use_rerank:
        rerank_input = selected_docs[:candidate_n] if candidate_n > 0 else []
        debug["rerank_candidate_count"] = len(rerank_input)

        if rerank_input and keep_n > 0:
            reranker = build_reranker(
                use_rerank=True,
                rerank_model_name=rerank_model_name,
                reranker_type=reranker_type,
                rerank_max_length=rerank_max_length,
                rerank_batch_size=rerank_batch_size,
            )
            reranked = reranker.rerank(query=query, docs=rerank_input)
            selected_docs = filter_reranked_docs(
                reranked_items=reranked,
                keep_n=keep_n,
                score_threshold=threshold,
            )
            debug["top_rerank_scores"] = [round(float(x.score), 4) for x in reranked[:3]]
        else:
            selected_docs = []
    else:
        # Backward-compatible default: preserve retrieved list unless a context cap is explicitly set.
        debug["rerank_candidate_count"] = len(selected_docs)

    if max_context_chunks and max_context_chunks > 0:
        selected_docs = selected_docs[: int(max_context_chunks)]

    debug["reranked_chunk_count"] = len(selected_docs)
    debug["fallback_no_context"] = len(selected_docs) == 0
    return selected_docs, debug


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
    output_token_logprobs: bool = False,
    logprob_last_k: int = 5,
) -> dict[str, Any]:
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

    if output_token_logprobs:
        generate_kwargs["return_dict_in_generate"] = True
        generate_kwargs["output_scores"] = True

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    generated_token_ids: list[int]
    token_logprobs: list[float] = []

    if output_token_logprobs and hasattr(outputs, "sequences"):
        sequences = outputs.sequences
        generated_ids = sequences[0][inputs["input_ids"].shape[-1] :]
        generated_token_ids = [int(x) for x in generated_ids.tolist()]

        if hasattr(outputs, "scores") and outputs.scores:
            transition_scores = model.compute_transition_scores(
                sequences=sequences,
                scores=outputs.scores,
                normalize_logits=True,
            )
            gen_steps = len(outputs.scores)
            token_logprobs = [float(transition_scores[0, i].item()) for i in range(gen_steps)]
            if len(token_logprobs) > len(generated_token_ids):
                token_logprobs = token_logprobs[: len(generated_token_ids)]
            elif len(token_logprobs) < len(generated_token_ids):
                generated_token_ids = generated_token_ids[: len(token_logprobs)]
    else:
        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        generated_token_ids = [int(x) for x in generated_ids.tolist()]

    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 简单清理，避免有些模型把 "Answer:" 又重复输出
    if text.lower().startswith("answer:"):
        text = text[len("answer:") :].strip()

    # 只保留第一行，减少 verbose 输出对 EM/F1 的伤害
    text = text.splitlines()[0].strip()

    logprob_features = compute_logprob_features(token_logprobs, last_k=logprob_last_k)

    return {
        "prediction_raw": text,
        "generated_token_ids": generated_token_ids,
        "token_logprobs": token_logprobs,
        "avg_logprob": logprob_features["avg_logprob"],
        "min_logprob": logprob_features["min_logprob"],
        "last_k_avg_logprob": logprob_features["last_k_avg_logprob"],
        "length_normalized_nll": logprob_features["length_normalized_nll"],
    }


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
    use_rerank: bool = False,
    rerank_model_name: str | None = None,
    rerank_top_n: int = 0,
    rerank_keep_n: int = 0,
    rerank_score_threshold: float | None = None,
    allow_ignore_context: bool = False,
    max_context_chunks: int = 0,
    log_rag_debug: bool = False,
    reranker_type: str = "cross_encoder",
    rerank_max_length: int = 512,
    rerank_batch_size: int = 16,
    output_token_logprobs: bool = False,
    logprob_last_k: int = 5,
    uncertainty_enabled: bool = False,
    uncertainty_score_type: str = "avg_logprob",
    uncertainty_score_weights: dict[str, float] | None = None,
    selective_enabled: bool = False,
    selective_threshold: float = -2.0,
    selective_ood_delta: float = 0.0,
    selective_refusal_text: str = "I'm not confident enough to answer this reliably.",
) -> dict[str, Any]:
    question = str(sample.get("question", ""))
    dataset = str(sample.get("dataset", ""))

    reference_fields = extract_reference_fields(sample)

    retrieved_docs: list[dict[str, Any]] = []
    retrieval_debug: dict[str, Any] | None = None
    if mode in {"rag", "sft_rag"}:
        retrieved_docs_raw = _normalize_retrieved_docs(
            retrieve(
                query=question,
                top_k=top_k,
                index_path=index_path,
                mapping_path=mapping_path,
                embedding_model_name=embedding_model_name,
            )
        )
        # Conservative defaults keep old behavior unless rerank/caps are explicitly enabled.
        effective_rerank_top_n = int(rerank_top_n) if rerank_top_n and rerank_top_n > 0 else int(top_k)
        effective_rerank_keep_n = int(rerank_keep_n) if rerank_keep_n and rerank_keep_n > 0 else int(top_k)
        retrieved_docs, retrieval_debug = _apply_rerank(
            query=question,
            retrieved_docs=retrieved_docs_raw,
            use_rerank=use_rerank,
            rerank_model_name=rerank_model_name,
            rerank_top_n=effective_rerank_top_n,
            rerank_keep_n=effective_rerank_keep_n,
            rerank_score_threshold=rerank_score_threshold,
            reranker_type=reranker_type,
            rerank_max_length=rerank_max_length,
            rerank_batch_size=rerank_batch_size,
            max_context_chunks=max_context_chunks,
        )

        if log_rag_debug and retrieval_debug is not None:
            print(
                "[rag_debug] "
                f"id={sample.get('id', '')} "
                f"retrieved={retrieval_debug['retrieved_chunk_count']} "
                f"reranked={retrieval_debug['reranked_chunk_count']} "
                f"top_scores={retrieval_debug['top_rerank_scores']} "
                f"fallback_no_context={retrieval_debug['fallback_no_context']}"
            )

    prompt = build_prompt(
        question=question,
        mode=mode,
        retrieved_docs=retrieved_docs,
        allow_ignore_context=allow_ignore_context,
    )
    generation = _generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_token_logprobs=output_token_logprobs,
        logprob_last_k=logprob_last_k,
    )
    prediction = str(generation["prediction_raw"])
    split_type = str(sample.get("split_type", "id")).lower()

    confidence_score: float | None = None
    if uncertainty_enabled:
        confidence_score = compute_confidence_score(
            row=generation,
            score_type=uncertainty_score_type,
            last_k=logprob_last_k,
            score_weights=uncertainty_score_weights,
        )

    threshold_used: float | None = None
    abstained = False
    abstain_reason = "selective_disabled"
    prediction_final = prediction
    if selective_enabled:
        threshold_used = threshold_for_split(
            base_threshold=float(selective_threshold),
            split_type=split_type,
            ood_delta=float(selective_ood_delta),
        )
        abstained, abstain_reason = apply_abstention(
            confidence_score=confidence_score,
            threshold=threshold_used,
        )
        if abstained:
            prediction_final = str(selective_refusal_text)

    return {
        "id": str(sample.get("id", "")),
        "dataset": dataset,
        "question": question,
        "split_type": split_type,
        **reference_fields,
        "mode": mode,
        "prediction_raw": prediction,
        # Backward-compatible field used by current eval scripts; now reflects final visible output.
        "prediction": prediction_final,
        "prediction_final": prediction_final,
        "generated_token_ids": generation["generated_token_ids"],
        "token_logprobs": generation["token_logprobs"],
        "avg_logprob": generation["avg_logprob"],
        "min_logprob": generation["min_logprob"],
        "last_k_avg_logprob": generation["last_k_avg_logprob"],
        "length_normalized_nll": generation["length_normalized_nll"],
        "confidence_score": confidence_score,
        "threshold_used": threshold_used,
        "abstained": abstained,
        "abstain_reason": abstain_reason,
        "retrieved_docs": retrieved_docs if mode in {"rag", "sft_rag"} else [],
        "retrieval_debug": retrieval_debug if mode in {"rag", "sft_rag"} else None,
    }


def save_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
