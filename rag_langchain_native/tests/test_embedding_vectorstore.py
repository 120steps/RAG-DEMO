from langchain_core.documents import Document

from rag_langchain_native.embedding import embedding_config
from rag_langchain_native.vectorstore import add_documents, as_retriever


def test_embedding_configuration_uses_distinct_e5_prefixes(v3_settings):
    config = embedding_config(v3_settings)
    assert config["model_name"] == "intfloat/multilingual-e5-base"
    assert config["query_prefix"] == "query: "
    assert config["document_prefix"] == "passage: "
    assert config["normalization"] is True
    assert config["dimension"] == 768


def test_chroma_vectorstore_and_as_retriever(fake_store):
    docs = [
        Document(
            page_content="国际出差需要总经理审批",
            metadata={
                "source": "travel.pdf",
                "page": 1,
                "chunk_id": 0,
                "document_id": "travel-1-0",
            },
        ),
        Document(
            page_content="密码每九十天更换",
            metadata={
                "source": "security.pdf",
                "page": 2,
                "chunk_id": 0,
                "document_id": "security-2-0",
            },
        ),
    ]
    add_documents(fake_store, docs)
    retriever = as_retriever(fake_store, 2)
    result = retriever.invoke("国际出差需要总经理审批")
    assert len(result) == 2
    assert all("document_id" in document.metadata for document in result)

