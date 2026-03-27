from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RerankItem:
    """Structured reranking output per candidate chunk."""

    doc: dict[str, Any]
    score: float
    rank: int


class BaseReranker(ABC):
    """Base interface for scoring (question, chunk) relevance."""

    @abstractmethod
    def score(self, query: str, docs: list[dict[str, Any]]) -> list[float]:
        """Return one relevance score per input doc, in the same order."""

    def rerank(self, query: str, docs: list[dict[str, Any]]) -> list[RerankItem]:
        if not docs:
            return []

        scores = self.score(query=query, docs=docs)
        if len(scores) != len(docs):
            raise ValueError("Reranker returned mismatched score length")

        ranked = sorted(
            (RerankItem(doc=doc, score=float(score), rank=0) for doc, score in zip(docs, scores)),
            key=lambda item: item.score,
            reverse=True,
        )
        for idx, item in enumerate(ranked, start=1):
            item.rank = idx
        return ranked


class NoOpReranker(BaseReranker):
    """No-op reranker that preserves original retrieval order."""

    def score(self, query: str, docs: list[dict[str, Any]]) -> list[float]:
        _ = query
        # Keep ordering: first item gets largest synthetic score.
        n = len(docs)
        return [float(n - i) for i in range(n)]


class CrossEncoderReranker(BaseReranker):
    """HF cross-encoder reranker based on sequence classification."""

    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        if not model_name:
            raise ValueError("rerank_model_name must be provided when use_rerank=true")

        self.model_name = model_name
        self.max_length = max(16, int(max_length))
        self.batch_size = max(1, int(batch_size))

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Cross-encoder reranking requires transformers and torch. "
                "Install missing dependencies before enabling use_rerank."
            ) from exc

        self._torch = torch
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
        except Exception as exc:  # pragma: no cover - runtime/model loading path
            raise RuntimeError(
                f"Failed to load reranker model '{model_name}'. "
                "Check rerank_model_name and local/network model availability."
            ) from exc

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()

    def _compute_scores(self, logits: Any) -> Any:
        torch = self._torch
        if logits.ndim == 1:
            return logits

        if logits.shape[-1] == 1:
            return torch.sigmoid(logits.squeeze(-1))

        probs = torch.softmax(logits, dim=-1)
        return probs[..., 1] if logits.shape[-1] > 1 else probs.squeeze(-1)

    def score(self, query: str, docs: list[dict[str, Any]]) -> list[float]:
        torch = self._torch
        if not docs:
            return []

        pairs = [(query, str(doc.get("text", ""))) for doc in docs]
        all_scores: list[float] = []

        with torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                q_batch = [x[0] for x in batch]
                p_batch = [x[1] for x in batch]

                inputs = self.tokenizer(
                    q_batch,
                    p_batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                batch_scores = self._compute_scores(outputs.logits)
                all_scores.extend(float(x) for x in batch_scores.detach().cpu().tolist())

        return all_scores


def build_reranker(
    use_rerank: bool,
    rerank_model_name: str | None,
    reranker_type: str = "cross_encoder",
    rerank_max_length: int = 512,
    rerank_batch_size: int = 16,
) -> BaseReranker:
    """Factory for reranker implementations."""

    if not use_rerank:
        return NoOpReranker()

    normalized_type = (reranker_type or "cross_encoder").strip().lower()
    if normalized_type not in {"cross_encoder", "cross-encoder"}:
        raise ValueError(f"Unsupported reranker_type: {reranker_type}")

    return CrossEncoderReranker(
        model_name=str(rerank_model_name or ""),
        max_length=rerank_max_length,
        batch_size=rerank_batch_size,
    )


def filter_reranked_docs(
    reranked_items: list[RerankItem],
    keep_n: int,
    score_threshold: float | None,
) -> list[dict[str, Any]]:
    """Filter reranked chunks by top-N and optional threshold."""

    if keep_n <= 0:
        return []

    selected: list[dict[str, Any]] = []
    threshold = float(score_threshold) if score_threshold is not None else None

    for item in reranked_items:
        if threshold is not None and item.score < threshold:
            continue

        doc = dict(item.doc)
        doc["rerank_score"] = float(item.score)
        doc["rerank_rank"] = int(item.rank)
        selected.append(doc)
        if len(selected) >= keep_n:
            break

    return selected
