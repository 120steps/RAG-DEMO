from __future__ import annotations

import hashlib
from dataclasses import replace

import fitz
import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from rag_langchain_native.config import DEFAULT_SETTINGS
from rag_langchain_native.catalog import DocumentCatalog
from rag_langchain_native.lifecycle import DocumentLifecycleService
from rag_langchain_native.security import AuthService


class DeterministicEmbeddings(Embeddings):
    dimension = 12

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [float(byte) / 255.0 for byte in digest[: cls.dimension]]
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        return [value / norm for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def v3_settings(tmp_path):
    runtime = tmp_path / "runtime"
    return replace(
        DEFAULT_SETTINGS,
        runtime_dir=runtime,
        chroma_dir=runtime / "chroma",
        cache_dir=runtime / "cache",
        upload_dir=runtime / "uploads",
        catalog_path=runtime / "catalog.sqlite3",
        collection_name="test_native",
        enterprise_collection_name="test_enterprise",
        auth_secret="pytest-enterprise-secret",
        reranker_enabled=False,
        rewrite_enabled=False,
        expansion_enabled=False,
        answerability_guard_enabled=False,
        vector_k=10,
        bm25_k=10,
        candidate_k=20,
        final_k=3,
        gemini_min_interval_seconds=0.0,
    )


@pytest.fixture
def fake_store(tmp_path):
    return Chroma(
        collection_name="test_collection",
        embedding_function=DeterministicEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
        collection_metadata={"hnsw:space": "cosine"},
    )


@pytest.fixture
def make_pdf_bytes():
    """创建只存在于内存的测试 PDF，不写入或修改仓库真实 PDF。"""
    def factory(text: str) -> bytes:
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), text)
        content = document.tobytes()
        document.close()
        return content

    return factory


@pytest.fixture
def enterprise_env(fake_store, v3_settings):
    catalog = DocumentCatalog(v3_settings.catalog_path)
    auth = AuthService(
        catalog,
        v3_settings.auth_secret,
        v3_settings.auth_token_ttl_seconds,
    )
    lifecycle = DocumentLifecycleService(
        catalog, v3_settings, vectorstore=fake_store
    )
    return {
        "catalog": catalog,
        "auth": auth,
        "lifecycle": lifecycle,
        "store": fake_store,
        "settings": v3_settings,
    }

