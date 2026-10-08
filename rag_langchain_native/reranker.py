"""Local LangChain cross-encoder document compressor."""

from __future__ import annotations

import operator
from functools import lru_cache
from typing import Sequence

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document

from .config import DEFAULT_SETTINGS, Settings
from .retrieval import clone_document


class ScoredCrossEncoderReranker(CrossEncoderReranker):
    """CrossEncoderReranker that retains the score in Document.metadata."""

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
        scores = self.model.score(
            [(query, document.page_content) for document in documents]
        )
        scored = []
        for document, score in zip(documents, scores, strict=True):
            copied = clone_document(document)
            copied.metadata["rerank_score"] = float(score)
            scored.append((copied, float(score)))
        scored.sort(key=operator.itemgetter(1), reverse=True)
        return [document for document, _ in scored[: self.top_n]]


@lru_cache(maxsize=2)
def _cached_reranker(
    model_name: str,
    device: str,
    cache_folder: str,
    top_n: int,
) -> ScoredCrossEncoderReranker:
    model = HuggingFaceCrossEncoder(
        model_name=model_name,
        model_kwargs={
            "device": device,
            "cache_folder": cache_folder,
            "max_length": 512,
        },
    )
    return ScoredCrossEncoderReranker(model=model, top_n=top_n)


def get_reranker(
    settings: Settings = DEFAULT_SETTINGS,
) -> ScoredCrossEncoderReranker:
    settings.ensure_runtime_dirs()
    model_cache = settings.cache_dir / "huggingface"
    model_cache.mkdir(parents=True, exist_ok=True)
    return _cached_reranker(
        settings.reranker_model,
        settings.embedding_device,
        str(model_cache),
        settings.final_k,
    )


def rerank_documents(
    query: str,
    documents: list[Document],
    *,
    settings: Settings = DEFAULT_SETTINGS,
    compressor: ScoredCrossEncoderReranker | None = None,
) -> list[Document]:
    if not settings.reranker_enabled:
        return documents[: settings.final_k]
    active = compressor or get_reranker(settings)
    return list(active.compress_documents(documents, query))

