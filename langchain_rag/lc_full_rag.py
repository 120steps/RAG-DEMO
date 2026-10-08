import time
from functools import lru_cache
from typing import Any

import chromadb
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)

from answer_guard import assess_answerability
from config import (
    ANSWERABILITY_GUARD_ENABLED,
    QUERY_EXPANSION_ENABLED,
    QUERY_REWRITE_ENABLED,
    REFUSAL_MESSAGE,
    TOP_K,
)
from langchain_rag.lc_llm import get_chat_model
from langchain_rag.lc_rag import RAG_PROMPT, strip_model_citations
from langchain_rag.lc_vectorstore import (
    SOURCE_CHROMA_DIR,
    SOURCE_COLLECTION_NAME,
)
from query_expander import expand_query
from query_rewriter import rewrite_query
from retrieval import retrieve_candidates


@lru_cache(maxsize=1)
def get_source_collection():
    client = chromadb.PersistentClient(path=str(SOURCE_CHROMA_DIR))
    return client.get_collection(name=SOURCE_COLLECTION_NAME)


def _prepare_queries(
    state: dict[str, Any],
    *,
    use_query_rewrite: bool,
    use_query_expansion: bool,
) -> dict[str, Any]:
    original_question = state["question"]
    retrieval_query = (
        rewrite_query(original_question)
        if use_query_rewrite
        else original_question
    )
    expanded_queries = (
        expand_query(retrieval_query)
        if use_query_expansion
        else [retrieval_query]
    )
    return {
        **state,
        "original_query": original_question,
        "rewritten_query": retrieval_query,
        "retrieval_query": retrieval_query,
        "expanded_queries": expanded_queries,
    }


def _retrieve(
    state: dict[str, Any],
    *,
    use_hybrid: bool,
    use_reranker: bool,
    use_query_expansion: bool,
    final_top_k: int,
) -> dict[str, Any]:
    candidates = retrieve_candidates(
        state["retrieval_query"],
        get_source_collection(),
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
        bm25_index=None,
        final_top_k=final_top_k,
        expanded_queries=(
            state["expanded_queries"]
            if use_query_expansion
            else None
        ),
    )
    retrieval_latency_ms = (
        time.perf_counter() - state["_request_started_at"]
    ) * 1000
    return {
        **state,
        "candidates": candidates,
        "answerability": assess_answerability(candidates),
        "retrieval_latency_ms": retrieval_latency_ms,
    }


def candidate_to_document(candidate: dict[str, Any]) -> Document:
    """Bind existing candidate data to a LangChain Document."""
    metadata = {
        **candidate["metadata"],
        "original_distance": candidate.get("original_distance"),
        "bm25_score": candidate.get("bm25_score"),
        "fusion_score": candidate.get("fusion_score"),
        "rerank_score": candidate.get("rerank_score"),
        "matched_queries": candidate.get("matched_queries", []),
    }
    return Document(
        page_content=candidate["document"],
        metadata=metadata,
    )


def format_candidate_context(candidates: list[dict[str, Any]]) -> str:
    context_parts = []
    for rank, candidate in enumerate(candidates, start=1):
        metadata = candidate["metadata"]
        context_parts.append(
            "\n".join(
                [
                    f"[Document {rank}]",
                    f"Source: {metadata.get('source', 'N/A')}",
                    f"Page: {metadata.get('page', 'N/A')}",
                    f"Chunk ID: {metadata.get('chunk_id', 'N/A')}",
                    "Content:",
                    candidate["document"],
                ]
            )
        )
    return "\n\n".join(context_parts)


def _add_context(state: dict[str, Any]) -> dict[str, Any]:
    return {
        **state,
        "context": format_candidate_context(state["candidates"]),
        "retrieved_documents": [
            candidate_to_document(candidate)
            for candidate in state["candidates"]
        ],
    }


def _should_refuse(state: dict[str, Any]) -> bool:
    return (
        ANSWERABILITY_GUARD_ENABLED
        and not state["answerability"]["should_answer"]
    )


def _build_refusal(state: dict[str, Any]) -> dict[str, Any]:
    return {
        **state,
        "answer": REFUSAL_MESSAGE,
        "refused": True,
        "generation_latency_ms": 0.0,
    }


def _start_generation(state: dict[str, Any]) -> dict[str, Any]:
    return {
        **state,
        "_generation_started_at": time.perf_counter(),
    }


def _finish_generation(state: dict[str, Any]) -> dict[str, Any]:
    return {
        **state,
        "refused": False,
        "generation_latency_ms": (
            time.perf_counter() - state["_generation_started_at"]
        )
        * 1000,
    }


def build_sources(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    seen = set()
    for candidate in candidates:
        metadata = candidate["metadata"]
        source = metadata.get("source")
        page = metadata.get("page")
        chunk_id = metadata.get("chunk_id")
        key = (source, page, chunk_id)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source": source,
                "page": page,
                "chunk_id": chunk_id,
            }
        )
    return sources


def serialize_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "document": candidate["document"],
            "metadata": candidate["metadata"],
            "original_distance": candidate.get("original_distance"),
            "distance": candidate.get("distance"),
            "bm25_score": candidate.get("bm25_score"),
            "fusion_score": candidate.get("fusion_score"),
            "rerank_score": candidate.get("rerank_score"),
            "matched_queries": candidate.get("matched_queries", []),
        }
        for rank, candidate in enumerate(candidates, start=1)
    ]


def _format_result(state: dict[str, Any]) -> dict[str, Any]:
    candidates = state["candidates"]
    total_latency_ms = (
        time.perf_counter() - state["_request_started_at"]
    ) * 1000
    return {
        "question": state["original_query"],
        "original_query": state["original_query"],
        "rewritten_query": state["rewritten_query"],
        "retrieval_query": state["retrieval_query"],
        "expanded_queries": state["expanded_queries"],
        "answer": state["answer"],
        "refused": state["refused"],
        "answerability": state["answerability"],
        "sources": build_sources(candidates),
        "documents": [candidate["document"] for candidate in candidates],
        "metadatas": [candidate["metadata"] for candidate in candidates],
        "distances": [
            candidate.get("original_distance") for candidate in candidates
        ],
        "bm25_scores": [
            candidate.get("bm25_score") for candidate in candidates
        ],
        "fusion_scores": [
            candidate.get("fusion_score") for candidate in candidates
        ],
        "rerank_scores": [
            candidate.get("rerank_score") for candidate in candidates
        ],
        "matched_queries": [
            candidate.get("matched_queries", [])
            for candidate in candidates
        ],
        "retrieved_documents": state["retrieved_documents"],
        "retrieved_context": serialize_candidates(candidates),
        "context": state["context"],
        "retrieval_latency_ms": state["retrieval_latency_ms"],
        "generation_latency_ms": state["generation_latency_ms"],
        "total_latency_ms": total_latency_ms,
    }


def build_advanced_retrieval_runnable(
    *,
    final_top_k: int = TOP_K,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    use_query_rewrite: bool | None = None,
    use_query_expansion: bool | None = None,
) -> Runnable:
    if final_top_k < 1:
        raise ValueError("final_top_k must be at least 1")

    rewrite_enabled = (
        QUERY_REWRITE_ENABLED
        if use_query_rewrite is None
        else use_query_rewrite
    )
    expansion_enabled = (
        QUERY_EXPANSION_ENABLED
        if use_query_expansion is None
        else use_query_expansion
    )

    return (
        RunnableLambda(
            lambda state: _prepare_queries(
                state,
                use_query_rewrite=rewrite_enabled,
                use_query_expansion=expansion_enabled,
            )
        )
        | RunnableLambda(
            lambda state: _retrieve(
                state,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
                use_query_expansion=expansion_enabled,
                final_top_k=final_top_k,
            )
        )
        | RunnableLambda(_add_context)
    )


def build_full_rag_chain(
    *,
    final_top_k: int = TOP_K,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    use_query_rewrite: bool | None = None,
    use_query_expansion: bool | None = None,
) -> Runnable:
    retrieval_runnable = build_advanced_retrieval_runnable(
        final_top_k=final_top_k,
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
        use_query_rewrite=use_query_rewrite,
        use_query_expansion=use_query_expansion,
    )
    answer_runnable = (
        RAG_PROMPT
        | get_chat_model()
        | StrOutputParser()
        | RunnableLambda(strip_model_citations)
    )
    generation_runnable = (
        RunnableLambda(_start_generation)
        | RunnablePassthrough.assign(answer=answer_runnable)
        | RunnableLambda(_finish_generation)
    )

    return (
        retrieval_runnable
        | RunnableBranch(
            (_should_refuse, RunnableLambda(_build_refusal)),
            generation_runnable,
        )
        | RunnableLambda(_format_result)
    )


def retrieve_advanced(
    question: str,
    *,
    final_top_k: int = TOP_K,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    use_query_rewrite: bool | None = None,
    use_query_expansion: bool | None = None,
) -> dict[str, Any]:
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    return build_advanced_retrieval_runnable(
        final_top_k=final_top_k,
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
        use_query_rewrite=use_query_rewrite,
        use_query_expansion=use_query_expansion,
    ).invoke(
        {
            "question": question.strip(),
            "_request_started_at": time.perf_counter(),
        }
    )


def ask_langchain_full_rag(
    question: str,
    *,
    final_top_k: int = TOP_K,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    use_query_rewrite: bool | None = None,
    use_query_expansion: bool | None = None,
) -> dict[str, Any]:
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    return build_full_rag_chain(
        final_top_k=final_top_k,
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
        use_query_rewrite=use_query_rewrite,
        use_query_expansion=use_query_expansion,
    ).invoke(
        {
            "question": question.strip(),
            "_request_started_at": time.perf_counter(),
        }
    )
