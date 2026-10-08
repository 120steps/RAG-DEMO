"""V3-only LangChain Chroma vector store."""

from __future__ import annotations

from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import DEFAULT_SETTINGS, Settings
from .embedding import get_embeddings


def create_vectorstore(settings: Settings = DEFAULT_SETTINGS) -> Chroma:
    settings.ensure_runtime_dirs()
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine", "owner": "v3"},
    )


def as_retriever(
    vectorstore: Chroma,
    top_k: int,
):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def get_all_documents(vectorstore: Chroma) -> list[Document]:
    result = vectorstore.get(include=["documents", "metadatas"])
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    ids = result.get("ids") or []
    return [
        Document(
            page_content=text,
            metadata={**(metadata or {}), "document_id": doc_id},
        )
        for doc_id, text, metadata in zip(ids, documents, metadatas)
    ]


def add_documents(
    vectorstore: Chroma,
    documents: Iterable[Document],
) -> list[str]:
    docs = list(documents)
    ids = [str(doc.metadata["document_id"]) for doc in docs]
    if docs:
        vectorstore.add_documents(docs, ids=ids)
    return ids

