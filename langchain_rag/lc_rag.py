from operator import itemgetter
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from config import REFUSAL_MESSAGE, TOP_K
from langchain_rag.lc_llm import get_chat_model
from langchain_rag.lc_retriever import as_retriever


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是企业知识库助手。

只能根据提供的知识库上下文回答问题，不允许使用知识库之外的信息。
如果上下文中没有足够信息，回答：{refusal_message}
回答必须简洁、直接，不要补充上下文未支持的事实。
引用由系统根据检索文档的 Metadata 生成；不要自行编造或输出来源。""",
        ),
        (
            "human",
            """知识库上下文：
{context}

问题：
{question}""",
        ),
    ]
).partial(refusal_message=REFUSAL_MESSAGE)

CITATION_LINE_PATTERN = re.compile(
    r"^\s*(?:[-*]\s*)?(?:来源|引用|参考来源|source|sources|citation|citations)\s*[:：]",
    re.IGNORECASE,
)


def format_context(documents: list[Document]) -> str:
    """Render retrieved documents as grounded context for the model."""
    context_parts = []
    for rank, document in enumerate(documents, start=1):
        metadata = document.metadata
        context_parts.append(
            "\n".join(
                [
                    f"[Document {rank}]",
                    f"Source: {metadata.get('source', 'N/A')}",
                    f"Page: {metadata.get('page', 'N/A')}",
                    f"Chunk ID: {metadata.get('chunk_id', 'N/A')}",
                    "Content:",
                    document.page_content,
                ]
            )
        )
    return "\n\n".join(context_parts)


def build_citations(documents: list[Document]) -> list[dict[str, Any]]:
    """Build deterministic citations exclusively from retrieved metadata."""
    citations = []
    seen = set()

    for document in documents:
        source = document.metadata.get("source")
        page = document.metadata.get("page")
        citation_key = (source, page)
        if citation_key in seen:
            continue

        seen.add(citation_key)
        citations.append(
            {
                "source": source,
                "page": page,
            }
        )

    return citations


def serialize_documents(
    documents: list[Document],
) -> list[dict[str, Any]]:
    return [
        {
            "id": document.id,
            "page_content": document.page_content,
            "metadata": dict(document.metadata),
        }
        for document in documents
    ]


def strip_model_citations(answer: str) -> str:
    """Discard model-authored citation lines; Metadata is authoritative."""
    lines = [
        line
        for line in answer.splitlines()
        if not CITATION_LINE_PATTERN.match(line)
    ]
    return "\n".join(lines).strip()


def _format_result(values: dict[str, Any]) -> dict[str, Any]:
    retrieved_documents = values["retrieved_documents"]
    return {
        "question": values["question"],
        "answer": values["answer"],
        "sources": build_citations(retrieved_documents),
        "documents": serialize_documents(retrieved_documents),
        "context": values["context"],
    }


def build_rag_chain(top_k: int = TOP_K) -> Runnable:
    """Build Question -> Retriever -> Prompt -> Gemini -> result Runnable."""
    retriever = as_retriever(top_k=top_k)

    retrieval_chain = RunnablePassthrough.assign(
        retrieved_documents=itemgetter("question") | retriever,
    )
    context_chain = retrieval_chain.assign(
        context=RunnableLambda(
            lambda values: format_context(values["retrieved_documents"])
        ),
    )
    answer_chain = context_chain.assign(
        answer=(
            RAG_PROMPT
            | get_chat_model()
            | StrOutputParser()
            | RunnableLambda(strip_model_citations)
        ),
    )
    return answer_chain | RunnableLambda(_format_result)


def ask_langchain_rag(
    question: str,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    return build_rag_chain(top_k=top_k).invoke(
        {"question": question.strip()}
    )
