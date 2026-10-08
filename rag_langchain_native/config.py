"""Configuration owned by the independent V3 application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    package_dir: Path = PACKAGE_DIR
    project_root: Path = PROJECT_ROOT
    source_pdf_dir: Path = PROJECT_ROOT / "data" / "pdf"
    runtime_dir: Path = PACKAGE_DIR / "runtime"
    chroma_dir: Path = PACKAGE_DIR / "runtime" / "chroma"
    cache_dir: Path = PACKAGE_DIR / "runtime" / "cache"
    upload_dir: Path = PACKAGE_DIR / "runtime" / "uploads"

    collection_name: str = os.getenv(
        "V3_CHROMA_COLLECTION", "company_knowledge_native"
    )
    chunk_size: int = int(os.getenv("V3_CHUNK_SIZE", "200"))
    chunk_overlap: int = int(os.getenv("V3_CHUNK_OVERLAP", "30"))

    embedding_model: str = os.getenv(
        "V3_EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
    )
    embedding_device: str = os.getenv("V3_EMBEDDING_DEVICE", "cpu")
    embedding_batch_size: int = int(
        os.getenv("V3_EMBEDDING_BATCH_SIZE", "32")
    )
    normalize_embeddings: bool = _bool_env(
        "V3_NORMALIZE_EMBEDDINGS", True
    )
    query_prefix: str = "query: "
    document_prefix: str = "passage: "

    vector_enabled: bool = _bool_env("V3_VECTOR_ENABLED", True)
    bm25_enabled: bool = _bool_env("V3_BM25_ENABLED", True)
    vector_k: int = int(os.getenv("V3_VECTOR_K", "10"))
    bm25_k: int = int(os.getenv("V3_BM25_K", "10"))
    candidate_k: int = int(os.getenv("V3_CANDIDATE_K", "20"))
    final_k: int = int(os.getenv("V3_FINAL_K", "3"))
    vector_weight: float = float(os.getenv("V3_VECTOR_WEIGHT", "0.5"))
    bm25_weight: float = float(os.getenv("V3_BM25_WEIGHT", "0.5"))
    rrf_k: int = int(os.getenv("V3_RRF_K", "60"))

    reranker_enabled: bool = _bool_env("V3_RERANKER_ENABLED", True)
    reranker_model: str = os.getenv(
        "V3_RERANKER_MODEL",
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    )

    rewrite_enabled: bool = _bool_env("V3_QUERY_REWRITE_ENABLED", False)
    expansion_enabled: bool = _bool_env(
        "V3_QUERY_EXPANSION_ENABLED", False
    )
    expansion_count: int = int(os.getenv("V3_QUERY_EXPANSION_COUNT", "3"))
    llm_model: str = os.getenv("V3_LLM_MODEL", "gemini-3.1-flash-lite")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    gemini_min_interval_seconds: float = float(
        os.getenv("V3_GEMINI_MIN_INTERVAL_SECONDS", "3.5")
    )

    answerability_guard_enabled: bool = _bool_env(
        "V3_ANSWERABILITY_GUARD_ENABLED", True
    )
    min_rerank_score: float = float(
        os.getenv("V3_MIN_RERANK_SCORE", "2.0")
    )
    min_bm25_score: float = float(os.getenv("V3_MIN_BM25_SCORE", "32.0"))
    max_vector_distance: float = float(
        os.getenv("V3_MAX_VECTOR_DISTANCE", "0.24")
    )
    refusal_message: str = "根据当前知识库无法回答该问题。"

    def ensure_runtime_dirs(self) -> None:
        for directory in (
            self.runtime_dir,
            self.chroma_dir,
            self.cache_dir,
            self.upload_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


DEFAULT_SETTINGS = Settings()

