"""V3 的本地 Cross-Encoder Reranker 模块。

文件职责：
    接收 Retrieval 产生的候选 ``list[Document]``，让本地 Cross-Encoder 同时阅读 Query
    与每个 Chunk，重新计算相关性并保留 Top-K。

在 RAG Pipeline 中的位置：
    Hybrid / Multi-query Retrieval 之后，Context Construction 之前。必须先用便宜的
    Retriever 缩小候选池，再运行更昂贵的 Cross-Encoder。

Embedding 与 Cross-Encoder 的区别：
    Embedding 分别编码 Query 和 Document，向量可预先存库，适合全库快速召回；
    Cross-Encoder 把 ``(query, document)`` 成对输入模型，交互更充分、排序更精细，但
    每个新 Query 都要重新计算每个 pair，因此不适合直接扫描全库。

输入与输出：
    输入为 ``query: str`` 和 ``documents: list[Document]``；输出是重新排序且截断的
    ``list[Document]``。source/page/chunk_id 等 Metadata 会保留，并新增 rerank_score。

LangChain 组件与边界：
    使用 ``HuggingFaceCrossEncoder`` 和 ``CrossEncoderReranker`` 接口。本项目没有使用
    ``ContextualCompressionRetriever``；而是先执行 Retrieval，再直接调用 compressor，
    以便主 LCEL 流程清晰保存候选和调试分数。
"""

from __future__ import annotations

import operator
from functools import lru_cache
from typing import Sequence

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document

from .config import DEFAULT_SETTINGS, Settings
from .observability import get_observability
from .retrieval import clone_document


class ScoredCrossEncoderReranker(CrossEncoderReranker):
    """在 Document Metadata 中保留分数的 CrossEncoderReranker。

    LangChain 原生 Reranker 能负责排序和截断，但项目的 Debug、Guard 和 Evaluation 还
    需要看到每个候选的实际 score。因此覆盖 ``compress_documents``，保留框架接口的
    同时增加 ``metadata["rerank_score"]``。
    """

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """对候选 Document 打分、排序并保留 ``top_n``。

        参数：
            documents (Sequence[Document]): Retriever 返回的候选。Sequence 表示列表、
                元组等有顺序的集合都可传入。
            query (str): 用于最终排序的单条 Retrieval Query。
            callbacks: 可选 LangChain Callback 配置；当前实现没有主动使用。

        返回：
            Sequence[Document]: 按 rerank_score 从高到低的 Top-N Document。

        执行过程：
            1. 空候选直接返回空列表。
            2. 列表推导式创建 ``(query, page_content)`` pairs。
            3. ``model.score`` 为每对文本生成相关性分数。
            4. 用 ``zip(..., strict=True)`` 绑定 Document 与 score，并克隆 Metadata。
            5. 降序排列后截取 ``self.top_n``。

        限制：
            Reranker 只能重排输入候选；正确 Chunk 没有被 Retriever 召回时，它无法凭空
            找回。因此 Retrieval Hit@K 和 Reranker 效果必须分别评估。
        """
        if not documents:
            return []
        # Cross-Encoder 的输入是文本对，不是预先计算好的单独向量。
        scores = self.model.score(
            [(query, document.page_content) for document in documents]
        )
        scored = []
        for document, score in zip(documents, scores, strict=True):
            copied = clone_document(document)
            copied.metadata["rerank_score"] = float(score)
            scored.append((copied, float(score)))
        scored.sort(key=operator.itemgetter(1), reverse=True)
        return [document for document, _ in scored[: self.top_n]]


@lru_cache(maxsize=2)
def _cached_reranker(
    model_name: str,
    device: str,
    cache_folder: str,
    top_n: int,
) -> ScoredCrossEncoderReranker:
    """按模型配置缓存本地 Reranker 与 compressor。

    ``@lru_cache(maxsize=2)`` 避免每次请求重新加载模型。maxsize=2 允许测试或消融中保留
    少量不同配置。``top_n`` 也属于缓存键，因为 compressor 本身保存该属性。
    """
    model = HuggingFaceCrossEncoder(
        model_name=model_name,
        model_kwargs={
            "device": device,
            "cache_folder": cache_folder,
            "max_length": 512,
        },
    )
    return ScoredCrossEncoderReranker(model=model, top_n=top_n)


def get_reranker(
    settings: Settings = DEFAULT_SETTINGS,
) -> ScoredCrossEncoderReranker:
    """根据 V3 Settings 获取缓存的本地 Cross-Encoder Reranker。

    参数：
        settings (Settings): 模型名、设备、Cache 目录与最终 Top-K。

    返回：
        ScoredCrossEncoderReranker: 可调用 ``compress_documents`` 的文档压缩器。

    调用关系：
        上游是 ``NativeRAGService.retrieve_only()`` 或 ``rerank_documents()``；下游是
        ``HuggingFaceCrossEncoder``。模型文件只缓存在 V3 runtime/cache。
    """
    settings.ensure_runtime_dirs()
    model_cache = settings.cache_dir / "huggingface"
    model_cache.mkdir(parents=True, exist_ok=True)
    return _cached_reranker(
        settings.reranker_model,
        settings.embedding_device,
        str(model_cache),
        settings.final_k,
    )


def rerank_documents(
    query: str,
    documents: list[Document],
    *,
    settings: Settings = DEFAULT_SETTINGS,
    compressor: ScoredCrossEncoderReranker | None = None,
) -> list[Document]:
    """根据开关统一执行 Rerank，或只截取最终 Top-K。

    参数：
        query (str): Reranker 使用的 Retrieval Query，不是 Expansion Query 列表。
        documents (list[Document]): Fusion 后的统一候选池。
        settings (Settings): Reranker 开关和 final_k。
        compressor: 可注入的 Reranker，测试时可替换为 Fake，生产默认本地模型。

    返回：
        list[Document]: 最终进入 Context 的 Document。

    执行过程：
        关闭 Reranker 时保持候选现有顺序并截断；开启时取得 compressor 并调用
        ``compress_documents``。所有 Metadata 都跟随 Document，不使用易错的平行列表。

    初学者知识点：
        ``compressor or get_reranker(settings)`` 表示优先使用调用方注入对象，否则创建
        默认对象。这种依赖注入让离线测试不必真的加载大型模型。
    """
    observability = get_observability()
    if not settings.reranker_enabled:
        output = documents[: settings.final_k]
        observability.metric("reranker.skipped.count")
        return output
    with observability.stage(
        "reranker",
        attributes={"input_count": len(documents), "top_k": settings.final_k},
    ) as trace_data:
        active = compressor or get_reranker(settings)
        output = list(active.compress_documents(documents, query))
        trace_data["input_count"] = len(documents)
        trace_data["result_count"] = len(output)
        observability.metric("reranker.candidate.count", len(documents))
        return output

