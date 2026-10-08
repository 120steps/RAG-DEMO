"""LangChain-first LCEL RAG pipeline."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from .config import DEFAULT_SETTINGS, Settings
from .query_processing import QueryProcessor
from .reranker import ScoredCrossEncoderReranker, get_reranker, rerank_documents
from .retrieval import NativeRetrievalEngine, serialize_document
from .vectorstore import create_vectorstore


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是企业知识库问答助手。只能依据给定 Context 回答。"
            "如果 Context 不足以支持答案，明确说明无法根据知识库回答。"
            "不得使用外部知识，不得编造事实或来源。直接、准确、简洁地回答中文问题。"
            "引用由系统根据检索元数据生成，因此不要在答案中自行编造 citation。"
            "如果 Context 无法支持问题的核心答案，refused 必须为 true；否则为 false。"
            "如果答案的实质是“文档未说明/未提供所问的具体事实或数值”，即使可以复述相邻制度，"
            "也必须将 refused 设为 true。",
        ),
        (
            "human",
            "Original Question:\n{question}\n\nRetrieved Context:\n{context}",
        ),
    ]
)


class AnswerOutput(BaseModel):
    answer: str = Field(description="Chinese answer or a concise refusal")
    refused: bool = Field(description="Whether context is insufficient")
    refusal_reason: str | None = Field(
        default=None,
        description="Why the answer was refused; null when answered",
    )


def get_chat_model(settings: Settings = DEFAULT_SETTINGS):
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.0,
        max_retries=2,
    )


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}, "
        f"Chunk: {doc.metadata.get('chunk_id')}]\n{doc.page_content}"
        for doc in documents
    )


def build_citations(documents: list[Document]) -> list[dict]:
    citations = []
    seen = set()
    for document in documents:
        key = (
            document.metadata.get("source"),
            document.metadata.get("page"),
            document.metadata.get("chunk_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "source": key[0],
                "page": key[1],
                "chunk_id": key[2],
                "document_id": document.metadata.get("document_id"),
            }
        )
    return citations


def assess_answerability(
    documents: list[Document], settings: Settings
) -> tuple[bool, str]:
    if not settings.answerability_guard_enabled:
        return bool(documents), "guard_disabled"
    if not documents:
        return False, "no_documents"
    for document in documents:
        metadata = document.metadata
        rerank_score = metadata.get("rerank_score")
        bm25_score = metadata.get("bm25_score")
        vector_distance = metadata.get("vector_distance")
        if rerank_score is not None and float(rerank_score) >= settings.min_rerank_score:
            return True, "rerank_score"
        if bm25_score is not None and float(bm25_score) >= settings.min_bm25_score:
            return True, "bm25_score"
        if vector_distance is not None and float(vector_distance) <= settings.max_vector_distance:
            return True, "vector_distance"
    return False, "below_relevance_threshold"


class NativeRAGService:
    def __init__(
        self,
        *,
        settings: Settings = DEFAULT_SETTINGS,
        vectorstore=None,
        llm: Runnable | None = None,
        query_model: Runnable | None = None,
        reranker=None,
    ) -> None:
        self.settings = settings
        self.vectorstore = vectorstore or create_vectorstore(settings)
        self.retrieval_engine = NativeRetrievalEngine(
            self.vectorstore, settings
        )
        self._llm = llm
        self._query_model = query_model
        self._reranker = reranker
        self._chain: Runnable | None = None

    def refresh_retriever(self) -> None:
        self.retrieval_engine = NativeRetrievalEngine(
            self.vectorstore, self.settings
        )
        self._chain = None

    def _model(self) -> Runnable:
        return self._llm or get_chat_model(self.settings)

    def _prepare_query_processor(self) -> QueryProcessor:
        query_model = self._query_model
        if query_model is None and (
            self.settings.rewrite_enabled or self.settings.expansion_enabled
        ):
            query_model = self._model()
        return QueryProcessor(query_model, self.settings)

    def _merge_query_state(self, value: dict) -> dict:
        return {
            "original_question": value["original_question"],
            **value["query_processing"],
        }

    def _retrieve(self, state: dict) -> dict:
        started = time.perf_counter()
        documents = self.retrieval_engine.retrieve_queries(
            state["expanded_queries"]
        )
        return {
            **state,
            "candidate_documents": documents,
            "retrieval_latency_ms": (time.perf_counter() - started) * 1000,
        }

    def _rerank(self, state: dict) -> dict:
        documents = rerank_documents(
            state["retrieval_query"],
            state["candidate_documents"],
            settings=self.settings,
            compressor=self._reranker,
        )
        return {**state, "retrieved_documents": documents}

    def _guard(self, state: dict) -> dict:
        allowed, reason = assess_answerability(
            state["retrieved_documents"], self.settings
        )
        return {
            **state,
            "context": format_context(state["retrieved_documents"]),
            "refused": not allowed,
            "refusal_reason": reason,
        }

    def _finalize(self, state: dict) -> dict:
        documents = state["retrieved_documents"]
        return {
            "question": state["original_question"],
            "original_query": state["original_question"],
            "retrieval_query": state["retrieval_query"],
            "expanded_queries": state["expanded_queries"],
            "rewrite_failed": state["rewrite_failed"],
            "expansion_failed": state["expansion_failed"],
            "answer": state["answer"],
            "refused": state["refused"],
            "refusal_reason": state["refusal_reason"],
            "sources": build_citations(documents),
            "documents": [
                serialize_document(document, rank)
                for rank, document in enumerate(documents, start=1)
            ],
            "retrieved_context": state["context"],
            "retrieval_latency_ms": state["retrieval_latency_ms"],
        }

    def _apply_answer_result(self, state: dict) -> dict:
        result = state["answer_result"]
        if isinstance(result, AnswerOutput):
            value = result
        elif isinstance(result, dict):
            value = AnswerOutput.model_validate(result)
        else:
            value = AnswerOutput(answer=str(result), refused=False)
        return {
            **state,
            "answer": value.answer,
            "refused": bool(value.refused),
            "refusal_reason": (
                value.refusal_reason
                if value.refused
                else state["refusal_reason"]
            ),
        }

    def build_rag_chain(self) -> Runnable:
        processor = self._prepare_query_processor()
        query_stage = RunnableParallel(
            original_question=RunnablePassthrough(),
            query_processing=processor.as_runnable(),
        ) | RunnableLambda(self._merge_query_state).with_config(
            run_name="merge_query_state"
        )

        answer_inputs = RunnableParallel(
            question=RunnableLambda(lambda state: state["original_question"]),
            context=RunnableLambda(lambda state: state["context"]),
        )
        model = self._model()
        if hasattr(model, "with_structured_output"):
            answer_model = model.with_structured_output(AnswerOutput)
            answer_chain = (
                answer_inputs | ANSWER_PROMPT | answer_model
            ).with_config(run_name="answer_generation")
        else:
            answer_chain = (
                answer_inputs
                | ANSWER_PROMPT
                | model
                | StrOutputParser()
                | RunnableLambda(
                    lambda text: AnswerOutput(answer=text, refused=False)
                )
            ).with_config(run_name="answer_generation_fallback")

        success = (
            RunnablePassthrough.assign(answer_result=answer_chain)
            | RunnableLambda(self._apply_answer_result)
            | RunnableLambda(self._finalize)
        )
        refusal = (
            RunnablePassthrough.assign(
                answer=lambda _: self.settings.refusal_message
            )
            | RunnableLambda(self._finalize)
        )

        return (
            RunnableLambda(lambda question: str(question).strip())
            | query_stage
            | RunnableLambda(self._retrieve).with_config(run_name="retrieval")
            | RunnableLambda(self._rerank).with_config(run_name="reranker")
            | RunnableLambda(self._guard).with_config(run_name="answer_guard")
            | RunnableBranch((lambda state: state["refused"], refusal), success)
        ).with_config(run_name="native_rag_v3")

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            self._chain = self.build_rag_chain()
        return self._chain

    def ask_rag(self, question: str) -> dict:
        return self.chain.invoke(question)

    async def aask_rag(self, question: str) -> dict:
        return await self.chain.ainvoke(question)

    def batch_ask(self, questions: list[str]) -> list[dict]:
        return self.chain.batch(questions)

    def retrieve_only(
        self,
        question: str,
        *,
        top_k: int = 10,
    ) -> dict:
        processor = self._prepare_query_processor()
        query_state = processor.process(question.strip())
        started = time.perf_counter()
        documents = self.retrieval_engine.retrieve_queries(
            query_state["expanded_queries"]
        )
        if self.settings.reranker_enabled:
            compressor = self._reranker or get_reranker(self.settings)
            if compressor.top_n != top_k:
                compressor = ScoredCrossEncoderReranker(
                    model=compressor.model,
                    top_n=top_k,
                )
            documents = rerank_documents(
                query_state["retrieval_query"],
                documents,
                settings=self.settings,
                compressor=compressor,
            )
        latency = (time.perf_counter() - started) * 1000
        return {
            "original_query": question,
            **query_state,
            "documents": [
                serialize_document(document, rank)
                for rank, document in enumerate(documents[:top_k], start=1)
            ],
            "latency_ms": latency,
        }


@lru_cache(maxsize=1)
def get_service() -> NativeRAGService:
    return NativeRAGService()


def build_rag_chain() -> Runnable:
    return get_service().chain


def ask_rag(question: str) -> dict:
    return get_service().ask_rag(question)

