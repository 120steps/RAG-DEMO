from langchain_classic.retrievers.document_compressors.cross_encoder import (
    BaseCrossEncoder,
)
from langchain_core.documents import Document

from rag_langchain_native.reranker import ScoredCrossEncoderReranker
from rag_langchain_native.retrieval import (
    NativeRetrievalEngine,
    chinese_tokenize,
    document_key,
)
from rag_langchain_native.vectorstore import add_documents


class KeywordCrossEncoder(BaseCrossEncoder):
    def score(self, text_pairs):
        return [2.0 if query in document else -1.0 for query, document in text_pairs]


def _documents():
    return [
        Document(
            page_content="MACsec 网络加密功能支持说明",
            metadata={"source": "s.pdf", "page": 1, "chunk_id": 0, "document_id": "a"},
        ),
        Document(
            page_content="国际出差审批流程与批准人",
            metadata={"source": "t.pdf", "page": 2, "chunk_id": 0, "document_id": "b"},
        ),
        Document(
            page_content="员工密码修改周期要求",
            metadata={"source": "e.pdf", "page": 3, "chunk_id": 0, "document_id": "c"},
        ),
    ]


def test_chinese_bm25_and_hybrid_rrf(fake_store, v3_settings):
    add_documents(fake_store, _documents())
    engine = NativeRetrievalEngine(fake_store, v3_settings)
    result = engine.retrieve("国际出差审批")
    assert result
    assert any("bm25" in doc.metadata.get("matched_retrievers", []) for doc in result)
    assert len({document_key(doc) for doc in result}) == len(result)
    assert chinese_tokenize("国际出差")


def test_multi_query_fusion_deduplicates_by_document_id(fake_store, v3_settings):
    add_documents(fake_store, _documents())
    engine = NativeRetrievalEngine(fake_store, v3_settings)
    result = engine.retrieve_queries(["国际出差审批", "出差批准人"])
    assert len({document_key(doc) for doc in result}) == len(result)
    assert all(doc.metadata.get("matched_queries") for doc in result)


def test_cross_encoder_reranker_keeps_metadata_and_score():
    reranker = ScoredCrossEncoderReranker(
        model=KeywordCrossEncoder(), top_n=2
    )
    docs = _documents()
    result = list(reranker.compress_documents(docs, "国际出差审批流程与批准人"))
    assert result[0].metadata["document_id"] == "b"
    assert result[0].metadata["rerank_score"] == 2.0

