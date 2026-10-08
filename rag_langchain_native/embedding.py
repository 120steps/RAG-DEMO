"""LangChain Hugging Face embeddings with E5 query/document semantics."""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from .config import DEFAULT_SETTINGS, Settings


@lru_cache(maxsize=2)
def _cached_embeddings(
    model_name: str,
    device: str,
    cache_folder: str,
    batch_size: int,
    normalize: bool,
    query_prefix: str,
    document_prefix: str,
) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model=model_name,
        cache_folder=cache_folder,
        model_kwargs={"device": device},
        encode_kwargs={
            "batch_size": batch_size,
            "normalize_embeddings": normalize,
            "prompt": document_prefix,
        },
        query_encode_kwargs={
            "batch_size": batch_size,
            "normalize_embeddings": normalize,
            "prompt": query_prefix,
        },
        show_progress=False,
    )


def get_embeddings(
    settings: Settings = DEFAULT_SETTINGS,
) -> HuggingFaceEmbeddings:
    settings.ensure_runtime_dirs()
    model_cache = settings.cache_dir / "huggingface"
    model_cache.mkdir(parents=True, exist_ok=True)
    return _cached_embeddings(
        settings.embedding_model,
        settings.embedding_device,
        str(model_cache),
        settings.embedding_batch_size,
        settings.normalize_embeddings,
        settings.query_prefix,
        settings.document_prefix,
    )


def embedding_config(settings: Settings = DEFAULT_SETTINGS) -> dict:
    return {
        "model_name": settings.embedding_model,
        "dimension": 768,
        "normalization": settings.normalize_embeddings,
        "query_prefix": settings.query_prefix,
        "document_prefix": settings.document_prefix,
        "device": settings.embedding_device,
        "batch_size": settings.embedding_batch_size,
    }

