"""LangChain-native vector, BM25, hybrid and multi-query retrieval."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import jieba
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from .config import DEFAULT_SETTINGS, Settings
from .vectorstore import get_all_documents


def chinese_tokenize(text: str) -> list[str]:
    return [token.strip().lower() for token in jieba.lcut(text) if token.strip()]


def document_key(document: Document) -> str:
    metadata = document.metadata
    document_id = metadata.get("document_id")
    if document_id:
        return str(document_id)
    return "|".join(
        str(metadata.get(name, ""))
        for name in ("source", "page", "chunk_id")
    )


def clone_document(document: Document) -> Document:
    return Document(
        page_content=document.page_content,
        metadata=dict(document.metadata),
    )


class ScoredChromaRetriever(BaseRetriever):
    """A Chroma retriever that keeps the cosine distance for debugging."""

    vectorstore: Any
    k: int = 10
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        results = self.vectorstore.similarity_search_with_score(query, k=self.k)
        documents = []
        for rank, (document, distance) in enumerate(results, start=1):
            copied = clone_document(document)
            copied.metadata.update(
                vector_distance=float(distance),
                vector_rank=rank,
            )
            documents.append(copied)
        return documents


class TracedEnsembleRetriever(EnsembleRetriever):
    """LangChain EnsembleRetriever with RRF trace metadata retained."""

    route_names: list[str] = Field(default_factory=list)

    def weighted_reciprocal_rank(
        self,
        doc_lists: list[list[Document]],
    ) -> list[Document]:
        if len(doc_lists) != len(self.weights):
            raise ValueError("Retriever lists and weights must have equal lengths")

        scores: dict[str, float] = defaultdict(float)
        first_seen: dict[str, Document] = {}
        matched_routes: dict[str, list[str]] = defaultdict(list)
        route_ranks: dict[str, dict[str, int]] = defaultdict(dict)

        for index, (documents, weight) in enumerate(
            zip(doc_lists, self.weights, strict=True)
        ):
            route = (
                self.route_names[index]
                if index < len(self.route_names)
                else f"retriever_{index + 1}"
            )
            seen: set[str] = set()
            for rank, document in enumerate(documents, start=1):
                key = document_key(document)
                if key in seen:
                    continue
                seen.add(key)
                scores[key] += float(weight) / (rank + self.c)
                matched_routes[key].append(route)
                route_ranks[key][route] = rank
                if key not in first_seen:
                    first_seen[key] = clone_document(document)
                else:
                    first_seen[key].metadata.update(document.metadata)

        output = []
        for key in sorted(scores, key=scores.get, reverse=True):
            document = first_seen[key]
            document.metadata.update(
                fusion_score=scores[key],
                matched_retrievers=matched_routes[key],
                retriever_ranks=route_ranks[key],
            )
            output.append(document)
        return output


def _rrf_queries(
    result_sets: list[list[Document]],
    queries: list[str],
    rrf_k: int,
) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, Document] = {}
    matched_queries: dict[str, list[str]] = defaultdict(list)
    query_ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for query, documents in zip(queries, result_sets, strict=True):
        seen: set[str] = set()
        for rank, document in enumerate(documents, start=1):
            key = document_key(document)
            if key in seen:
                continue
            seen.add(key)
            scores[key] += 1.0 / (rrf_k + rank)
            matched_queries[key].append(query)
            query_ranks[key][query] = rank
            if key not in first_seen:
                first_seen[key] = clone_document(document)

    output = []
    for key in sorted(scores, key=scores.get, reverse=True):
        document = first_seen[key]
        if "fusion_score" in document.metadata:
            document.metadata["hybrid_fusion_score"] = document.metadata[
                "fusion_score"
            ]
        document.metadata.update(
            fusion_score=scores[key],
            matched_queries=matched_queries[key],
            query_ranks=query_ranks[key],
        )
        output.append(document)
    return output


class NativeRetrievalEngine:
    def __init__(
        self,
        vectorstore,
        settings: Settings = DEFAULT_SETTINGS,
    ) -> None:
        self.vectorstore = vectorstore
        self.settings = settings
        self.documents = get_all_documents(vectorstore)
        self.vector_retriever = ScoredChromaRetriever(
            vectorstore=vectorstore,
            k=settings.vector_k,
        )
        self.bm25_retriever = self._build_bm25()
        self.base_retriever = self._build_base_retriever()

    def _build_bm25(self) -> BM25Retriever | None:
        if not self.documents:
            return None
        return BM25Retriever.from_documents(
            self.documents,
            preprocess_func=chinese_tokenize,
            k=self.settings.bm25_k,
        )

    def _build_base_retriever(self):
        retrievers = []
        weights = []
        names = []
        if self.settings.vector_enabled:
            retrievers.append(self.vector_retriever)
            weights.append(self.settings.vector_weight)
            names.append("vector")
        if self.settings.bm25_enabled and self.bm25_retriever is not None:
            retrievers.append(self.bm25_retriever)
            weights.append(self.settings.bm25_weight)
            names.append("bm25")
        if not retrievers:
            raise ValueError("At least one retrieval route must be enabled")
        if len(retrievers) == 1:
            return retrievers[0]
        return TracedEnsembleRetriever(
            retrievers=retrievers,
            weights=weights,
            c=self.settings.rrf_k,
            id_key="document_id",
            route_names=names,
        )

    def retrieve(self, query: str) -> list[Document]:
        documents = self.base_retriever.invoke(query)
        return list(documents[: self.settings.candidate_k])

    def retrieve_queries(self, queries: list[str]) -> list[Document]:
        unique_queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
        if not unique_queries:
            return []
        result_sets = self.base_retriever.batch(unique_queries)
        if len(unique_queries) == 1:
            documents = list(result_sets[0])
            for document in documents:
                document.metadata.setdefault("matched_queries", unique_queries)
            return documents[: self.settings.candidate_k]
        fused = _rrf_queries(result_sets, unique_queries, self.settings.rrf_k)
        return fused[: self.settings.candidate_k]

    def debug(self, query: str, top_k: int | None = None) -> list[dict]:
        limit = top_k or self.settings.final_k
        return [serialize_document(doc, rank) for rank, doc in enumerate(
            self.retrieve(query)[:limit], start=1
        )]


def serialize_document(document: Document, rank: int | None = None) -> dict:
    metadata = dict(document.metadata)
    return {
        "rank": rank,
        "document": document.page_content,
        "source": metadata.get("source"),
        "page": metadata.get("page"),
        "chunk_id": metadata.get("chunk_id"),
        "document_id": metadata.get("document_id"),
        "vector_distance": metadata.get("vector_distance"),
        "fusion_score": metadata.get("fusion_score"),
        "rerank_score": metadata.get("rerank_score"),
        "matched_retrievers": metadata.get("matched_retrievers", []),
        "matched_queries": metadata.get("matched_queries", []),
        "metadata": metadata,
    }

