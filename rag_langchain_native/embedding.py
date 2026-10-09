"""V3 的 LangChain Embedding 模块。

文件职责：
    创建并缓存 ``HuggingFaceEmbeddings``，把文本转换为 768 维浮点向量。

在 RAG Pipeline 中的位置：
    Ingestion 时，Chroma 调用 ``embed_documents()`` 为 Chunk 生成文档向量；Query 时，
    Chroma 调用 ``embed_query()`` 为检索问题生成 Query 向量。两类向量必须来自兼容的
    同一模型空间，向量距离才有意义。

输入与输出：
    输入是字符串或字符串列表；LangChain Embeddings 接口输出 ``list[float]`` 或
    ``list[list[float]]``。本文件对外主要返回 Embeddings 对象，而不直接执行检索。

为什么需要 Embedding：
    Embedding 模型把文本语义编码成数字坐标。语义相近的 Query 与 Chunk 往往在向量
    空间中更接近，因此即使用词不完全相同也可能被 Vector Search 找到。

LangChain 价值：
    V1 需要分别维护 ``embed_text`` 和 ``embed_texts``。V3 通过统一 Embeddings 接口
    让 Chroma 自动选择 Query/Document 编码入口，同时保留 E5 所需的不同前缀。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from .config import DEFAULT_SETTINGS, Settings


@lru_cache(maxsize=2)
def _cached_embeddings(
    model_name: str,
    device: str,
    cache_folder: str,
    batch_size: int,
    normalize: bool,
    query_prefix: str,
    document_prefix: str,
) -> HuggingFaceEmbeddings:
    """按完整配置创建并缓存 Hugging Face Embeddings 对象。

    参数分别描述模型名、运行设备、缓存目录、批量大小、是否归一化，以及 Query 和
    Document 的 E5 前缀。返回值实现 LangChain ``Embeddings`` 接口。

    执行过程：
        1. ``encode_kwargs`` 配置文档编码，即 ``embed_documents()``。
        2. ``query_encode_kwargs`` 配置问题编码，即 ``embed_query()``。
        3. ``normalize_embeddings=True`` 将向量缩放为单位长度，便于使用 cosine 距离。

    初学者知识点：
        ``@lru_cache`` 是装饰器。相同参数再次调用时会复用已经加载的模型，避免每个
        请求都重新加载大型模型。函数名前的下划线表示它是模块内部辅助函数。
    """
    # E5 模型用 ``passage: `` 标明这是待检索文档，用 ``query: `` 标明这是搜索问题。
    # 两者仍进入同一个兼容向量空间，但不能把 Document 编码配置误用于 Query。
    return HuggingFaceEmbeddings(
        model=model_name,
        cache_folder=cache_folder,
        model_kwargs={"device": device},
        encode_kwargs={
            "batch_size": batch_size,
            "normalize_embeddings": normalize,
            "prompt": document_prefix,
        },
        query_encode_kwargs={
            "batch_size": batch_size,
            "normalize_embeddings": normalize,
            "prompt": query_prefix,
        },
        show_progress=False,
    )


def get_embeddings(
    settings: Settings = DEFAULT_SETTINGS,
) -> HuggingFaceEmbeddings:
    """取得供 V3 Chroma 使用的 Embeddings 对象。

    参数：
        settings (Settings): V3 配置，包含模型、设备、Batch Size、归一化和前缀。

    返回：
        HuggingFaceEmbeddings: LangChain Embeddings 实现。常用方法为
        ``embed_query(text)`` 和 ``embed_documents(texts)``。

    调用关系：
        上游是 ``vectorstore.create_vectorstore()``；下游是缓存工厂
        ``_cached_embeddings()``。Chroma 在入库或查询时再实际调用编码方法。
    """
    settings.ensure_runtime_dirs()
    model_cache = settings.cache_dir / "huggingface"
    model_cache.mkdir(parents=True, exist_ok=True)
    return _cached_embeddings(
        settings.embedding_model,
        settings.embedding_device,
        str(model_cache),
        settings.embedding_batch_size,
        settings.normalize_embeddings,
        settings.query_prefix,
        settings.document_prefix,
    )


def embedding_config(settings: Settings = DEFAULT_SETTINGS) -> dict:
    """把关键 Embedding 配置序列化为普通字典，供 Evaluation 记录。

    返回值不是向量，而是可写入 JSON 的配置快照。维度 768 对应当前
    ``intfloat/multilingual-e5-base``；如果以后更换模型，必须同步核对该字段。

    初学者知识点：
        ``-> dict`` 是返回类型注解；字典通过键名说明每个值的含义。
    """
    return {
        "model_name": settings.embedding_model,
        "dimension": 768,
        "normalization": settings.normalize_embeddings,
        "query_prefix": settings.query_prefix,
        "document_prefix": settings.document_prefix,
        "device": settings.embedding_device,
        "batch_size": settings.embedding_batch_size,
    }

