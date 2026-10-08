import asyncio
from dataclasses import replace

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag_langchain_native.chain import (
    NativeRAGService,
    assess_answerability,
    build_citations,
)
from rag_langchain_native.vectorstore import add_documents


def _fake_llm():
    return RunnableLambda(lambda _: AIMessage(content="需要总经理审批。"))


def test_lcel_pipeline_invoke_batch_and_ainvoke(fake_store, v3_settings):
    add_documents(
        fake_store,
        [
            Document(
                page_content="国际出差需要总经理审批。",
                metadata={
                    "source": "travel.pdf",
                    "page": 1,
                    "chunk_id": 0,
                    "document_id": "travel-1-0",
                },
            )
        ],
    )
    service = NativeRAGService(
        settings=v3_settings,
        vectorstore=fake_store,
        llm=_fake_llm(),
    )
    result = service.ask_rag("国际出差谁审批？")
    assert result["answer"] == "需要总经理审批。"
    assert result["question"] == "国际出差谁审批？"
    assert result["sources"][0]["source"] == "travel.pdf"
    assert len(service.batch_ask(["问题一", "问题二"])) == 2


def test_ainvoke(fake_store, v3_settings):
    add_documents(
        fake_store,
        [Document(page_content="制度", metadata={"source": "a.pdf", "page": 1, "chunk_id": 0, "document_id": "a"})],
    )
    service = NativeRAGService(settings=v3_settings, vectorstore=fake_store, llm=_fake_llm())
    result = asyncio.run(service.aask_rag("制度？"))
    assert result["answer"]


def test_citation_is_metadata_driven_and_refusal_is_structured(v3_settings):
    document = Document(
        page_content="内容",
        metadata={"source": "x.pdf", "page": 4, "chunk_id": 2},
    )
    assert build_citations([document]) == [
        {"source": "x.pdf", "page": 4, "chunk_id": 2, "document_id": None}
    ]
    strict = replace(v3_settings, answerability_guard_enabled=True)
    allowed, reason = assess_answerability([], strict)
    assert allowed is False
    assert reason == "no_documents"

