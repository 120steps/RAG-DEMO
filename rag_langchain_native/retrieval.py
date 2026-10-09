"""V3 的 LangChain Native Retrieval 模块。

文件职责：
    在 Query Processing 之后、Reranker 和 Context Construction 之前，完成候选 Chunk
    召回。支持 Vector Search、BM25、Hybrid RRF，以及多 Query Retrieval。

各组件职责：
    - Vector Search：根据 Embedding 语义相似度找候选，适合同义表达。
    - BM25：根据关键词、词频和稀有程度找候选，适合数字、缩写和专有名词。
    - Hybrid Search：融合两条路线，降低单一检索方法的盲区。
    - RRF：按名次而不是直接相加不同量纲的原始分数。
    - Reranker：不在本文件执行；本文件只扩大候选召回，之后由 ``reranker.py`` 精排。

输入与输出：
    输入是一条 ``str`` Query 或 ``list[str]`` Expansion Queries；输出是
    ``list[Document]``。每个 Document 的 ``page_content`` 是 Chunk，``metadata`` 保留
    source/page/chunk_id/document_id，以及 distance、rank、fusion trace 等调试字段。

调用关系：
    上游是 ``NativeRAGService._retrieve()`` 和 ``retrieve_only()``；下游是 LangChain
    Chroma、``BM25Retriever``、``EnsembleRetriever``。结果随后进入 Cross-Encoder。

LangChain 价值与自定义边界：
    Vector、BM25 和 Ensemble 都遵循 Retriever 协议，可使用 ``invoke()`` / ``batch()``。
    但为了保留原始 distance、融合路线和分数，本模块实现了两个最小子类；多 Query 的
    RRF 也仍是自定义逻辑，而不是使用 ``MultiQueryRetriever``。
"""

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
from .observability import get_observability
from .vectorstore import get_all_documents


def chinese_tokenize(text: str) -> list[str]:
    """使用 jieba 将中文文本切成适合 BM25 的词项。

    参数：
        text (str): Query 或 Chunk 文本。

    返回：
        list[str]: 去除空白并转为小写的词项列表。

    为什么需要：
        英文可以天然按空格分词，中文句子通常没有空格。如果把整句视为一个词，BM25
        很难匹配局部关键词。jieba 是分词工具，不负责向量语义。

    初学者知识点：
        这是列表推导式：先遍历 ``jieba.lcut`` 的结果，再过滤空词并转换大小写。
    """
    return [token.strip().lower() for token in jieba.lcut(text) if token.strip()]


def document_key(document: Document) -> str:
    """返回同一个 Chunk 在 Fusion 中使用的稳定去重键。

    优先使用 Ingestion 生成的 ``document_id``；如果旧数据没有该字段，则退回
    ``source|page|chunk_id``。不能只按文本去重，因为不同文件或页面可能有相同文本。

    参数：
        document (Document): 待识别的 LangChain Document。

    返回：
        str: 用于字典和集合的唯一键。
    """
    metadata = document.metadata
    # Phase 9 中 document_id 是“逻辑文档”，不能拿它给 Chunk 去重，否则同一 PDF 的
    # 所有页面都会被误合并。chunk_uid 才是 tenant/version/page/chunk 共同决定的唯一键。
    chunk_uid = metadata.get("chunk_uid")
    if chunk_uid:
        return str(chunk_uid)
    document_id = metadata.get("document_id")
    if document_id:
        return str(document_id)
    return "|".join(
        str(metadata.get(name, ""))
        for name in ("source", "page", "chunk_id")
    )


def clone_document(document: Document) -> Document:
    """复制 Document 的文本和 Metadata，避免融合时原地污染共享对象。

    ``dict(document.metadata)`` 创建浅拷贝；随后给副本增加 distance、rank 或 score，
    不会把调试字段意外写回 VectorStore/BM25 共用的原对象。
    """
    return Document(
        page_content=document.page_content,
        metadata=dict(document.metadata),
    )


class ScoredChromaRetriever(BaseRetriever):
    """能够保留 Chroma cosine distance 的 LangChain Retriever。

    为什么不直接只用 ``vectorstore.as_retriever()``：
        标准 Retriever 很方便，但通常只返回 Document。Evaluation 和 Answer Guard 还
        需要原始 distance，因此本类继承 ``BaseRetriever``，复用 LangChain Retriever
        生命周期，同时补充 score Metadata。

    属性：
        vectorstore: LangChain Chroma；Pydantic 需要允许任意第三方对象类型。
        k (int): Vector Search 最多返回多少个候选。

    返回契约：
        ``invoke(query)`` 最终调用 ``_get_relevant_documents``，返回 ``list[Document]``。
    """

    vectorstore: Any
    k: int = 10
    metadata_filter: dict[str, Any] | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """执行一次带 distance 的 Chroma 相似度搜索。

        参数：
            query (str): 检索 Query。
            run_manager: LangChain 注入的 Callback Manager。本实现没有主动调用它，但
                方法签名必须符合 ``BaseRetriever`` 协议，便于 tracing/callback 扩展。

        返回：
            list[Document]: 按 Chroma 相似度排序；Metadata 新增 ``vector_distance`` 和
            从 1 开始的 ``vector_rank``。

        执行过程：
            ``similarity_search_with_score`` 内部调用 Embeddings 的 ``embed_query()``，
            再查询 Chroma。返回元素是 ``(Document, distance)`` 二元组；本方法将它们
            重新绑定到同一个 Document 副本，防止文本、Metadata 与分数错位。

        初学者知识点：
            方法名以下划线开头是 LangChain BaseRetriever 要求实现的内部扩展点；用户
            通常调用公开的 ``invoke()``，而不是直接调用本方法。
        """
        observability = get_observability()
        with observability.stage(
            "vector.search", attributes={"top_k": self.k}
        ) as trace_data:
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=self.k,
                filter=self.metadata_filter,
            )
            documents = []
            for rank, (document, distance) in enumerate(results, start=1):
                copied = clone_document(document)
                copied.metadata.update(
                    vector_distance=float(distance),
                    vector_rank=rank,
                )
                documents.append(copied)
            trace_data["result_count"] = len(documents)
            observability.metric("retrieval.candidate.count", len(documents), attributes={"route": "vector"})
            return documents


class ObservedBM25Retriever(BM25Retriever):
    """为 LangChain BM25Retriever 增加安全 Span，不改变其排序算法。"""

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        observability = get_observability()
        with observability.stage("bm25.search", attributes={"top_k": self.k}) as trace_data:
            documents = list(super()._get_relevant_documents(query, run_manager=run_manager))
            trace_data["result_count"] = len(documents)
            observability.metric("retrieval.candidate.count", len(documents), attributes={"route": "bm25"})
            return documents


class TracedEnsembleRetriever(EnsembleRetriever):
    """保留 RRF 调试信息的 LangChain EnsembleRetriever。

    ``EnsembleRetriever`` 用统一 Retriever 接口调用 Vector 与 BM25，再进行 Reciprocal
    Rank Fusion。V3 覆盖融合方法，不是为了伪装成框架原生算法，而是为了额外保留：
    ``fusion_score``、命中的路线和每条路线排名。这些字段供调试和 Evaluation 使用。

    ``route_names`` 是 Pydantic Field；通常值为 ``["vector", "bm25"]``。
    """

    route_names: list[str] = Field(default_factory=list)

    def weighted_reciprocal_rank(
        self,
        doc_lists: list[list[Document]],
    ) -> list[Document]:
        """对多条 Retriever 排名执行加权 RRF，并按稳定 ID 去重。

        参数：
            doc_lists (list[list[Document]]): 每个 Retriever 的有序候选列表。

        返回：
            list[Document]: 按 RRF 分数降序排列的去重 Document。

        原理：
            每条路线中排名为 ``rank`` 的候选贡献 ``weight / (rank + c)``。Vector
            distance 与 BM25 内部分数的方向、尺度不同，直接相加没有统一意义；RRF 只
            依赖排名，因此更稳健。权重表达对各路线的相对信任，不是概率。

        执行过程：
            1. 验证候选列表数与权重数一致。
            2. 每条路线内部按稳定 key 去重。
            3. 同一 Chunk 多路线命中时累计分数并合并 Metadata。
            4. 写入 trace 字段并按 fusion score 排序。

        初学者知识点：
            ``defaultdict`` 在键首次访问时自动创建默认值；``set`` 用于快速判重；
            ``zip(..., strict=True)`` 在长度不等时立即报错，避免静默丢数据。
    """
        observability = get_observability()
        with observability.stage(
            "fusion.rrf",
            attributes={"route_count": len(doc_lists)},
        ) as trace_data:
            output = self._weighted_reciprocal_rank_impl(doc_lists)
            trace_data["result_count"] = len(output)
            return output

    def _weighted_reciprocal_rank_impl(
        self, doc_lists: list[list[Document]]
    ) -> list[Document]:
        """保留原 RRF 实现；外层方法只负责 Observability。"""
        if len(doc_lists) != len(self.weights):
            raise ValueError("Retriever lists and weights must have equal lengths")

        scores: dict[str, float] = defaultdict(float)
        first_seen: dict[str, Document] = {}
        matched_routes: dict[str, list[str]] = defaultdict(list)
        route_ranks: dict[str, dict[str, int]] = defaultdict(dict)

        # ``enumerate`` 同时提供路线编号和 ``(documents, weight)`` 值。
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
                # RRF 的核心公式：同一 Chunk 被多条路线找到时会累计贡献。
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
    """融合多条 Expansion Query 的检索结果。

    这与上面的 Hybrid RRF 是两个不同层次：Hybrid RRF 融合 Vector/BM25 路线；本函数
    融合“同一需求的不同 Query 表达”。同一个 Chunk 被多条 Query 找到时只返回一份，
    但会累加排名贡献并记录 ``matched_queries`` / ``query_ranks``。

    参数：
        result_sets: 与 queries 一一对应的候选列表。
        queries: 去重后的 Retrieval Query。
        rrf_k: RRF 平滑常数。

    返回：
        list[Document]: Query-level RRF 排序后的候选。

    注意：
        当前没有使用 ``MultiQueryRetriever``。其 Union 行为不能直接满足本项目对 RRF
        score、命中 Query 和排名 trace 的要求，所以保留了这段确定性 Python 逻辑。
    """
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
            # 所有 Expansion Query 当前使用相同权重，因此分子固定为 1.0。
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
    """组装并执行 V3 Retriever 的门面类。

    构造时读取 Chroma 中的全部 Document 建立内存 BM25，并根据开关选择 Vector、BM25
    或 Hybrid Retriever。在线阶段提供单 Query 和多 Query 两种入口。

    ``class`` 定义一种对象类型；``self`` 指当前 Engine 实例，因此多个方法可以共享
    vectorstore、settings 和已经建立的 Retriever，而不用每次重新传参。
    """

    def __init__(
        self,
        vectorstore,
        settings: Settings = DEFAULT_SETTINGS,
        *,
        metadata_filter: dict[str, Any] | None = None,
        allowed_chunk_ids: set[str] | None = None,
        allowed_version_ids: set[str] | None = None,
    ) -> None:
        """初始化 Vector、BM25 与最终 base_retriever。

        参数：
            vectorstore: 已连接的 LangChain Chroma。
            settings (Settings): Retrieval 开关、Top-K、权重和 RRF 常数。
            metadata_filter: 服务端生成的 Chroma tenant/kb/version Filter。
            allowed_chunk_ids: 可选的精确 Chunk 白名单。
            allowed_version_ids: Vector 与 BM25 共用的 active/authorized 版本集合；显式空集
                表示拒绝全部，绝不能解释为“没有过滤条件”。

        注意：
            ``get_all_documents`` 会把全库文本载入内存以构建 BM25。新增 PDF 后，API 会
            调 ``NativeRAGService.refresh_retriever()`` 重建本对象，使 BM25 看到新文档。
        """
        self.vectorstore = vectorstore
        self.settings = settings
        self.metadata_filter = metadata_filter
        self.allowed_chunk_ids = allowed_chunk_ids
        self.allowed_version_ids = allowed_version_ids
        self.deny_all = (
            allowed_chunk_ids is not None and not allowed_chunk_ids
        ) or (allowed_version_ids is not None and not allowed_version_ids)
        all_documents = get_all_documents(vectorstore)
        # BM25 是内存索引，无法让 Chroma 替它过滤。因此在建立 BM25 之前使用与 Vector
        # Search 相同的授权 chunk_uid 集合，确保两条召回路线拥有完全相同的安全边界。
        self.documents = (
            [
                document
                for document in all_documents
                if str(document.metadata.get("chunk_uid")) in allowed_chunk_ids
            ]
            if allowed_chunk_ids is not None
            else all_documents
        )
        if allowed_version_ids is not None:
            self.documents = [
                document
                for document in self.documents
                if str(document.metadata.get("version_id")) in allowed_version_ids
            ]
        self.vector_retriever = ScoredChromaRetriever(
            vectorstore=vectorstore,
            k=settings.vector_k,
            metadata_filter=metadata_filter,
        )
        self.bm25_retriever = self._build_bm25()
        self.base_retriever = self._build_base_retriever()

    def _build_bm25(self) -> BM25Retriever | None:
        """根据全部 Document 建立 LangChain BM25Retriever。

        返回 ``BM25Retriever`` 或 None。当前 Retriever 保留 Document Metadata，但不会
        自动把原始 BM25 数值写入 Metadata；Hybrid trace 能看到 route/rank/RRF score，
        不能把 ``bm25_score`` 当作当前已经稳定提供的字段。
        """
        if not self.documents:
            return None
        observability = get_observability()
        with observability.stage("bm25.index") as trace_data:
            retriever = ObservedBM25Retriever.from_documents(
                self.documents,
                preprocess_func=chinese_tokenize,
                k=self.settings.bm25_k,
            )
            trace_data["document_count"] = len(self.documents)
            return retriever

    def _build_base_retriever(self):
        """按配置组合本次实际使用的 Retriever。

        只开启一条路线时直接返回该 Retriever；两条路线都开启时返回
        ``TracedEnsembleRetriever``。至少一条路线必须启用，否则没有任何候选来源。

        ``id_key="document_id"`` 告诉 Ensemble 哪个 Metadata 字段表示同一 Chunk。
        ``weights`` 分别来自 vector_weight 与 bm25_weight。
        """
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
            id_key="chunk_uid",
            route_names=names,
        )

    def retrieve(self, query: str) -> list[Document]:
        """执行单条 Query Retrieval，并限制候选池大小。

        ``self.base_retriever.invoke(query)`` 是 LangChain Retriever 的标准同步调用：输入
        ``str``，输出有序 ``list[Document]``。这里只负责候选召回，不执行 Reranker。
        """
        if self.deny_all:
            return []
        observability = get_observability()
        with observability.stage(
            "retrieval", attributes={"query_count": 1, "top_k": self.settings.candidate_k}
        ) as trace_data:
            documents = self.base_retriever.invoke(
                query, config=observability.langchain_config()
            )
            output = list(documents[: self.settings.candidate_k])
            trace_data["result_count"] = len(output)
            return output

    def retrieve_queries(self, queries: list[str]) -> list[Document]:
        """对多条 Query 批量检索，并在需要时进行 Query-level RRF。

        参数：
            queries (list[str]): 至少包含原始 Retrieval Query，也可能包含 Expansion。

        返回：
            list[Document]: 最多 candidate_k 个融合候选，随后统一进入 Reranker。

        执行过程：
            1. ``dict.fromkeys`` 保序去重并过滤空 Query。
            2. ``base_retriever.batch(unique_queries)`` 使用统一批量接口执行每条 Query。
            3. 单 Query 不需要第二层融合，只记录 matched_queries。
            4. 多 Query 调 ``_rrf_queries``，而不是对每条 Query 分别 Rerank。

        ``batch()`` 与 ``invoke()`` 的区别：
            invoke 处理一个输入；batch 接收输入列表并返回对应结果列表。LangChain 可按
            Runnable 实现和配置调度批量工作，但不能假设所有底层一定并行或没有限流。
        """
        # ``dict`` 在现代 Python 中保留插入顺序，因此这种去重不会打乱 Query 顺序。
        unique_queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
        if not unique_queries or self.deny_all:
            return []
        observability = get_observability()
        with observability.stage(
            "retrieval",
            attributes={"query_count": len(unique_queries), "top_k": self.settings.candidate_k},
        ) as trace_data:
            result_sets = self.base_retriever.batch(
                unique_queries,
                config=observability.langchain_config(),
            )
            output = self._merge_query_results(unique_queries, result_sets)
            trace_data["result_count"] = len(output)
            return output

    def _merge_query_results(
        self, unique_queries: list[str], result_sets: list[list[Document]]
    ) -> list[Document]:
        """合并批量 Query 结果；从 retrieve_queries 拆出只为保持 Span 边界清晰。"""
        if len(unique_queries) == 1:
            documents = list(result_sets[0])
            for document in documents:
                document.metadata.setdefault("matched_queries", unique_queries)
            return documents[: self.settings.candidate_k]
        fused = _rrf_queries(result_sets, unique_queries, self.settings.rrf_k)
        return fused[: self.settings.candidate_k]

    def debug(self, query: str, top_k: int | None = None) -> list[dict]:
        """将单 Query Retrieval 结果序列化成便于查看的字典。

        该方法不调用 Gemini、不构造 Answer，只用于 Retrieval-only 调试。
        ``top_k or settings.final_k`` 表示未传值时采用最终返回数量。
        """
        limit = top_k or self.settings.final_k
        return [serialize_document(doc, rank) for rank, doc in enumerate(
            self.retrieve(query)[:limit], start=1
        )]


def serialize_document(document: Document, rank: int | None = None) -> dict:
    """把 LangChain Document 转成 API/Evaluation 可 JSON 序列化的字典。

    参数：
        document (Document): 检索或重排后的 Chunk。
        rank (int | None): 可选的最终排名。

    返回：
        dict: 包含文本、Citation Metadata 和各阶段调试分数。

    为什么需要转换：
        ``Document`` 是 Python/Pydantic 对象，API 和结果 JSON 更适合使用基本类型。
        同时保留完整 ``metadata``，避免只暴露少数字段后丢失实验信息。
    """
    metadata = dict(document.metadata)
    return {
        "rank": rank,
        "document": document.page_content,
        "source": metadata.get("source"),
        "page": metadata.get("page"),
        "chunk_id": metadata.get("chunk_id"),
        "document_id": metadata.get("document_id"),
        "version_id": metadata.get("version_id"),
        "tenant_id": metadata.get("tenant_id"),
        "knowledge_base_id": metadata.get("knowledge_base_id"),
        "classification": metadata.get("classification"),
        "chunk_uid": metadata.get("chunk_uid"),
        "vector_distance": metadata.get("vector_distance"),
        "fusion_score": metadata.get("fusion_score"),
        "rerank_score": metadata.get("rerank_score"),
        "matched_retrievers": metadata.get("matched_retrievers", []),
        "matched_queries": metadata.get("matched_queries", []),
        "metadata": metadata,
    }

