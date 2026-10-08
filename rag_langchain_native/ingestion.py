"""Native PDF ingestion: PyMuPDFLoader -> Documents -> splitter -> Chroma."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DEFAULT_SETTINGS, Settings
from .vectorstore import add_documents, create_vectorstore


def _stable_id(source: str, page: int, chunk_id: int) -> str:
    raw = f"{source}|{page}|{chunk_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def load_pdf_pages(pdf_path: str | Path) -> list[Document]:
    path = Path(pdf_path).resolve()
    pages = PyMuPDFLoader(str(path), mode="page").load()
    normalized = []
    for page in pages:
        metadata = dict(page.metadata)
        zero_based_page = int(metadata.get("page", 0))
        metadata.update(
            source=path.name,
            page=zero_based_page + 1,
        )
        normalized.append(
            Document(page_content=page.page_content, metadata=metadata)
        )
    return normalized


def split_pages(
    pages: list[Document],
    settings: Settings = DEFAULT_SETTINGS,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks: list[Document] = []
    for page in pages:
        page_chunks = splitter.split_documents([page])
        for index, chunk in enumerate(page_chunks):
            metadata = dict(chunk.metadata)
            source = str(metadata["source"])
            page_number = int(metadata["page"])
            metadata.update(
                chunk_id=index,
                document_id=_stable_id(source, page_number, index),
            )
            chunks.append(
                Document(
                    page_content=chunk.page_content,
                    metadata=metadata,
                )
            )
    return chunks


def ingest_pdf(
    pdf_path: str | Path,
    *,
    settings: Settings = DEFAULT_SETTINGS,
    vectorstore=None,
) -> dict:
    store = vectorstore or create_vectorstore(settings)
    pages = load_pdf_pages(pdf_path)
    chunks = split_pages(pages, settings)
    source = Path(pdf_path).name
    store.delete(where={"source": source})
    ids = add_documents(store, chunks)
    return {
        "source": source,
        "pages": len(pages),
        "chunks": len(chunks),
        "document_ids": ids,
    }


def _safe_clear_chroma(settings: Settings) -> None:
    target = settings.chroma_dir.resolve()
    runtime = settings.runtime_dir.resolve()
    if not target.is_relative_to(runtime) or target == runtime:
        raise ValueError(f"Refusing to delete non-V3 path: {target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def rebuild_knowledge_base(
    pdf_dir: str | Path | None = None,
    *,
    settings: Settings = DEFAULT_SETTINGS,
) -> dict:
    _safe_clear_chroma(settings)
    store = create_vectorstore(settings)
    directory = Path(pdf_dir or settings.source_pdf_dir)
    results = [
        ingest_pdf(path, settings=settings, vectorstore=store)
        for path in sorted(directory.glob("*.pdf"))
    ]
    return {
        "files": len(results),
        "chunks": sum(item["chunks"] for item in results),
        "details": results,
        "collection_count": store._collection.count(),
    }

