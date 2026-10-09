"""V3 专用的 LangChain Chroma VectorStore 模块。

文件职责：
    创建 V3 独立 Chroma、把 ``Document`` 写入向量库、读取全部 Document，并提供标准
    Retriever 工厂。数据库仅位于 ``rag_langchain_native/runtime/chroma``。

在 RAG Pipeline 中的位置：
    Ingestion 的终点是 ``add_documents()``；在线 Retrieval 从这里获得 Chroma。

重要数据结构：
    LangChain ``Document`` 同时保存：
    - ``page_content``：Chunk 文本；
    - ``metadata``：source、page、chunk_id、document_id 等；
    - ``id``：LangChain Document 可选字段。本项目稳定 ID 主要保存在 Metadata，并在
      ``add_documents(..., ids=ids)`` 时作为 Chroma ID 写入。

为什么 Metadata 不能丢：
    Retrieval、Reranker 和 Citation 都必须知道文本来自哪个文件、页码和 Chunk。
    如果把文本与 Metadata 分成容易错位的平行列表，排序后可能引用到错误页面。

LangChain 组件：
    ``Chroma`` 是 VectorStore；``as_retriever()`` 把它适配为统一的
    ``query -> list[Document]`` Retriever 接口。
"""

from __future__ import annotations

from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import DEFAULT_SETTINGS, Settings
from .embedding import get_embeddings


def create_vectorstore(settings: Settings = DEFAULT_SETTINGS) -> Chroma:
    """创建或连接 V3 自己的 LangChain Chroma。

    参数：
        settings (Settings): 提供 collection 名、持久化目录和 Embeddings 配置。

    返回：
        Chroma: LangChain VectorStore 对象。它封装了底层 Chroma collection。

    执行过程：
        1. 确保 V3 runtime 目录存在。
        2. 绑定 ``get_embeddings()``，让入库和查询自动调用正确编码方法。
        3. 使用 cosine 距离；``owner=v3`` 用于标记该 collection 的归属。

    调用关系：
        上游是 Ingestion 和 ``NativeRAGService``；下游是 LangChain Chroma。
    """
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
    """把 Chroma VectorStore 转换成标准相似度 Retriever。

    参数：
        vectorstore (Chroma): 已连接的 V3 向量库。
        top_k (int): 每次最多返回的相关 Document 数量。

    返回：
        VectorStoreRetriever: 可执行 ``invoke(query)`` 的 Retriever。其输入是 ``str``，
        输出是 ``list[Document]``。

    ``search_kwargs={"k": top_k}`` 会传给底层相似度搜索。V1 需要手动生成 Query
    Embedding、调用 ``collection.query``、解析 documents/metadatas；标准 Retriever
    可封装这些步骤。

    注意：
        V3 高级主链为了保留原始 distance，实际使用 ``ScoredChromaRetriever``。
        本函数是标准 Retriever 入口，不表示一次请求会同时运行两种 Vector Retriever。
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def get_all_documents(vectorstore: Chroma) -> list[Document]:
    """从 V3 Chroma 读取全部文本与 Metadata，并恢复为 Document 列表。

    参数：
        vectorstore (Chroma): V3 向量库。

    返回：
        list[Document]: 供 BM25 建索引的全部 Chunk。每个 Chroma ID 会写回
        ``metadata["document_id"]``，使 Hybrid/RRF 能稳定去重。

    执行过程：
        ``vectorstore.get()`` 返回多个平行列表；``zip`` 按相同位置重新绑定 ID、文本和
        Metadata。这里读取全库是为了建立内存 BM25，不是在线 Vector Search。
    """
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
    """把一组 LangChain Document 写入 V3 Chroma。

    参数：
        vectorstore (Chroma): 目标向量库。
        documents (Iterable[Document]): 可迭代的 Chunk Document。

    返回：
        list[str]: 实际使用的稳定 Chroma ID。

    执行过程：
        先转成 ``list``，因为后续既要遍历 ID，又要把同一批对象交给 Chroma；再从
        Metadata 取得 ``document_id``，调用 ``add_documents``。Chroma 会通过已绑定的
        Embeddings 自动执行 ``embed_documents()``。

    初学者知识点：
        ``Iterable`` 表示参数可以是列表、元组或生成器；转成 list 后才能安全重复遍历。
    """
    docs = list(documents)
    ids = [str(doc.metadata["document_id"]) for doc in docs]
    if docs:
        vectorstore.add_documents(docs, ids=ids)
    return ids

