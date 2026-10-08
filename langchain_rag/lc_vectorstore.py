from functools import lru_cache
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from langchain_rag.lc_embedding import ExistingEmbeddingAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHROMA_DIR = PROJECT_ROOT / "chroma_db"
LANGCHAIN_CHROMA_DIR = PROJECT_ROOT / "chroma_db_langchain"
SOURCE_COLLECTION_NAME = "company_knowledge"
LANGCHAIN_COLLECTION_NAME = "company_knowledge_langchain"
DISTANCE_SPACE = "l2"


def load_source_documents():
    """Read the handwritten RAG collection without mutating it."""
    source_client = chromadb.PersistentClient(
        path=str(SOURCE_CHROMA_DIR)
    )
    source_collection = source_client.get_collection(
        name=SOURCE_COLLECTION_NAME
    )
    source_data = source_collection.get(
        include=["documents", "metadatas"],
    )

    records = sorted(
        zip(
            source_data["ids"],
            source_data["documents"],
            source_data["metadatas"],
        ),
        key=lambda record: record[0],
    )
    documents = [
        Document(
            id=record_id,
            page_content=document,
            metadata=metadata,
        )
        for record_id, document, metadata in records
    ]
    ids = [record_id for record_id, _, _ in records]
    return documents, ids


@lru_cache(maxsize=1)
def get_vectorstore():
    return Chroma(
        collection_name=LANGCHAIN_COLLECTION_NAME,
        embedding_function=ExistingEmbeddingAdapter(),
        persist_directory=str(LANGCHAIN_CHROMA_DIR),
        collection_configuration={
            "hnsw": {"space": DISTANCE_SPACE}
        },
    )


def _validate_existing_collection(
    source_documents,
    source_ids,
    existing_data,
):
    source_by_id = {
        document_id: document
        for document_id, document in zip(source_ids, source_documents)
    }
    existing_by_id = {
        document_id: (page_content, metadata)
        for document_id, page_content, metadata in zip(
            existing_data["ids"],
            existing_data["documents"],
            existing_data["metadatas"],
        )
    }

    if set(source_by_id) != set(existing_by_id):
        raise RuntimeError(
            "LangChain collection IDs do not match the source collection. "
            "Refusing to modify the existing experiment database."
        )

    for document_id, source_document in source_by_id.items():
        existing_text, existing_metadata = existing_by_id[document_id]
        if (
            source_document.page_content != existing_text
            or source_document.metadata != existing_metadata
        ):
            raise RuntimeError(
                "LangChain collection content differs from the source "
                f"collection at ID {document_id!r}. Refusing to overwrite it."
            )


def initialize_vectorstore():
    """Create or validate the independent LangChain collection."""
    vectorstore = get_vectorstore()
    source_documents, source_ids = load_source_documents()
    existing_data = vectorstore.get(
        include=["documents", "metadatas"],
    )

    if not existing_data["ids"]:
        vectorstore.add_documents(
            documents=source_documents,
            ids=source_ids,
        )
    else:
        _validate_existing_collection(
            source_documents,
            source_ids,
            existing_data,
        )

    return vectorstore


def similarity_search(query: str, top_k: int = 5):
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    return initialize_vectorstore().similarity_search(query, k=top_k)


def similarity_search_with_distance(query: str, top_k: int = 5):
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    return initialize_vectorstore().similarity_search_with_score(
        query,
        k=top_k,
    )

