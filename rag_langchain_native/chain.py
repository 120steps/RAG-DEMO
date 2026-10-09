"""V3 的 LangChain-first Answer Pipeline 与业务服务入口。

文件职责：
    把已经独立实现的 Query Processing、Hybrid Retrieval、Reranker、Answer Guard、Context
    Construction、Prompt、Gemini、Structured Output 和 Citation 编排成一条 LCEL Chain。

完整数据流：
    Original Question
    -> Rewrite / Expansion（可选）
    -> Hybrid / Multi-query Retrieval
    -> Cross-Encoder Reranker
    -> Answer Guard
    -> Context + Original Question
    -> ChatPromptTemplate -> Gemini -> AnswerOutput
    -> Metadata Citation -> 最终 dict

关键数据类型：
    ``question`` 是 str；候选是 ``list[Document]``；``context`` 是 str；Prompt 执行后是
    ChatPromptValue/messages；ChatModel 通常返回 AIMessage或结构化 ``AnswerOutput``；最终
    API 结果是可 JSON 序列化的 dict。

为什么使用 LCEL：
    Runnable 统一提供 ``invoke``、``batch``、``ainvoke`` 等执行协议；``|`` 表示把前一
    Runnable 输出交给下一 Runnable；Parallel 保留多路数据，Branch 表达拒答/回答分支。
    但 Retrieval 算法、阈值、Context 格式和 Citation 规则仍是项目自己的 Python 逻辑。

当前没有实现的能力：
    本文件没有调用 ``stream()``，API 也没有 SSE ``StreamingResponse``，因此当前 V3
    不能宣称支持真正的 token streaming。``ainvoke`` 返回异步的完整结果，不等于流式。
"""

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
    """Gemini Answer Generation 的结构化输出模型。

    字段：
        answer: 中文答案或简短拒答文本。
        refused: 模型是否认为 Context 不足。
        refusal_reason: 拒答原因；正常回答时可以为 None。

    为什么需要：
        若只检查固定中文拒答句，模型换一种说法就难以判断。结构化 ``refused: bool``
        给 API 和 Evaluation 一个稳定字段。确定性 Answer Guard 会先做第一层判断，模型
        仍可在 Context 表面相关但无法支持核心答案时进行第二层拒答。
    """

    answer: str = Field(description="Chinese answer or a concise refusal")
    refused: bool = Field(description="Whether context is insufficient")
    refusal_reason: str | None = Field(
        default=None,
        description="Why the answer was refused; null when answered",
    )


def get_chat_model(settings: Settings = DEFAULT_SETTINGS):
    """根据 V3 配置创建 LangChain Gemini ChatModel。

    参数：
        settings (Settings): 模型名、API Key 等配置。

    返回：
        ChatGoogleGenerativeAI: 实现 Runnable 的 ChatModel，可接收 PromptValue/messages，
        返回 AIMessage，也支持 ``with_structured_output``。

    执行过程：
        先检查 Key 是否存在，再以 temperature=0 和有限重试创建模型。API Key 只从环境
        配置读取，不写入源码。

    LangChain 价值：
        V1 直接调用特定 SDK；V3 ChatModel 可直接放进 LCEL，并共享 invoke/ainvoke/
        batch、Structured Output 和 Callback 协议。它不会让底层 Gemini 本身更聪明。
    """
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.0,
        max_retries=2,
    )


def format_context(documents: list[Document]) -> str:
    """把最终 Retrieved Documents 格式化为 Gemini 可阅读的 Context 字符串。

    参数：
        documents (list[Document]): Reranker 之后的 Top-K 文档。

    返回：
        str: 每个 Chunk 前带 source、page、chunk_id 的连续文本。

    为什么需要：
        ChatModel 不会自动读取 Python Document 对象；必须把证据转换成 Prompt 中的文本。
        同时保留 Metadata 标签，有助于调试，但最终 Citation 仍由系统确定生成，不依赖
        LLM 自己输出页码。

    初学者知识点：
        生成器表达式逐个产生字符串，``"\n\n".join(...)`` 把它们连接成一个 Context。
    """
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}, "
        f"Chunk: {doc.metadata.get('chunk_id')}, "
        f"Version: {doc.metadata.get('version_id')}]\n{doc.page_content}"
        for doc in documents
    )


def build_citations(documents: list[Document]) -> list[dict]:
    """只根据 Retrieved Document Metadata 构建确定性 Citation。

    参数：
        documents (list[Document]): 最终送入 Context 的 Document。

    返回：
        list[dict]: 每项含 source、page、chunk_id、document_id。

    执行过程：
        用 ``(source, page, chunk_id)`` 元组作为去重键；第一次出现时创建 Citation。
        不只按文本去重，因为不同页可能有相同文字。

    为什么不能让 LLM 生成 Citation：
        LLM 输出的文件名/页码只是文本，可能编造。Metadata 来源于真正检索结果，才能
        稳定验证 ``expected_source + expected_page``。但 Citation 正确仍不等于答案中
        每个主张都忠实，需要 Faithfulness Evaluation。
    """
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
                "version_id": document.metadata.get("version_id"),
                "tenant_id": document.metadata.get("tenant_id"),
                "knowledge_base_id": document.metadata.get("knowledge_base_id"),
            }
        )
    return citations


def assess_answerability(
    documents: list[Document], settings: Settings
) -> tuple[bool, str]:
    """用确定性检索信号判断是否允许进入 Answer Generation。

    参数：
        documents (list[Document]): Reranker 后的最终候选。
        settings (Settings): Guard 开关和各分数阈值。

    返回：
        tuple[bool, str]: 是否允许回答，以及通过/拒绝的原因。

    执行过程：
        Guard 关闭时只要求有文档；无文档立即拒答；否则遍历候选，只要任一 Document 的
        rerank score、BM25 score 或 vector distance 达到阈值就允许回答。

    重要限制：
        当前 ``BM25Retriever`` 没有把原始 BM25 数值写入 Metadata，所以 bm25_score
        分支通常没有数据。默认开启 Reranker 时主要使用 rerank_score；没有 Reranker
        的 Vector 候选可使用 vector_distance。阈值必须通过拒答案例评估，而非凭感觉。
    """
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
    """V3 的业务服务对象，负责持有组件并构建/执行完整 LCEL Chain。

    该类不依赖 V1/V2 业务函数。构造器允许注入 VectorStore、LLM、Query Model 和
    Reranker，便于 pytest 使用 Fake 组件进行离线测试。

    初学者知识点：
        类把“数据”和“操作这些数据的方法”放在一起；``self`` 是当前服务实例。
        ``self._chain`` 以下划线开头，表示仅供类内部使用的缓存属性。
    """

    def __init__(
        self,
        *,
        settings: Settings = DEFAULT_SETTINGS,
        vectorstore=None,
        llm: Runnable | None = None,
        query_model: Runnable | None = None,
        reranker=None,
        metadata_filter: dict | None = None,
        allowed_version_ids: set[str] | None = None,
        cache_namespace: str = "default",
    ) -> None:
        """初始化 V3 Service 和 Retrieval Engine。

        参数：
            settings (Settings): 全部 V3 配置。
            vectorstore: 可选 Chroma/Fake Store；None 时连接 V3 Chroma。
            llm (Runnable | None): 可选 Answer Model，测试可注入 Fake。
            query_model (Runnable | None): 可选 Rewrite/Expansion Model。
            reranker: 可选 Cross-Encoder/Fake compressor。
            metadata_filter: Phase 9 服务端授权 Chroma Filter。
            allowed_version_ids: 同时约束内存 BM25 的授权版本集合。
            cache_namespace: tenant/kb/user/epoch 相关的 Query Cache 作用域。

        返回：
            None。构造完成后通过 ``ask_rag``、``retrieve_only`` 等方法使用。

        ``Runnable | None`` 中的 ``|`` 是类型联合；与 LCEL 表达式中的管道不是一回事。
        ``vectorstore or create_vectorstore`` 表示优先使用注入对象，否则创建默认对象。
        """
        self.settings = settings
        self.vectorstore = vectorstore or create_vectorstore(settings)
        self.metadata_filter = metadata_filter
        self.allowed_version_ids = allowed_version_ids
        self.cache_namespace = cache_namespace
        self.retrieval_engine = NativeRetrievalEngine(
            self.vectorstore,
            settings,
            metadata_filter=metadata_filter,
            allowed_version_ids=allowed_version_ids,
        )
        self._llm = llm
        self._query_model = query_model
        self._reranker = reranker
        self._chain: Runnable | None = None

    def refresh_retriever(self) -> None:
        """在知识库变化后重建 Retrieval Engine 并清除已构建 Chain。

        FastAPI 上传新 PDF 后，Chroma 已更新，但内存 BM25 仍保存旧文档集合。因此必须
        重建 Engine。把 ``_chain`` 设为 None，让下次访问 property 时使用新 Engine
        重新构建 LCEL 图。
        """
        self.retrieval_engine = NativeRetrievalEngine(
            self.vectorstore,
            self.settings,
            metadata_filter=self.metadata_filter,
            allowed_version_ids=self.allowed_version_ids,
        )
        self._chain = None

    def _model(self) -> Runnable:
        """返回注入的模型，或延迟创建真实 Gemini ChatModel。"""
        return self._llm or get_chat_model(self.settings)

    def _prepare_query_processor(self) -> QueryProcessor:
        """根据开关准备 QueryProcessor，避免关闭时无意义创建 Gemini。

        若调用方没有注入 query_model，且 Rewrite/Expansion 至少一项开启，才复用 Answer
        Model。两项都关闭时 QueryProcessor 接收 None，但其快速返回路径不需要模型。
        """
        query_model = self._query_model
        if query_model is None and (
            self.settings.rewrite_enabled or self.settings.expansion_enabled
        ):
            query_model = self._model()
        return QueryProcessor(
            query_model,
            self.settings,
            cache_namespace=self.cache_namespace,
        )

    @staticmethod
    def _normalize_chain_input(value: str | dict) -> dict:
        """兼容 Phase 8 字符串与 Phase 9 original/retrieval 双 Query 输入。"""
        if isinstance(value, dict):
            original = str(value.get("original_question", "")).strip()
            retrieval = str(value.get("retrieval_question") or original).strip()
        else:
            original = str(value).strip()
            retrieval = original
        if not original:
            raise ValueError("Question cannot be empty")
        return {"original_question": original, "retrieval_question": retrieval}

    def _merge_query_state(self, value: dict) -> dict:
        """合并 RunnableParallel 的两路输出。

        输入结构示例：
            ``{"original_question": str, "query_processing": dict}``

        返回结构：
            ``{"original_question": str, "retrieval_query": str,
            "expanded_queries": list[str], ...}``

        ``**value["query_processing"]`` 是字典展开，把内部键值复制到外层新字典。
        """
        return {
            "original_question": value["original_question"],
            **value["query_processing"],
        }

    def _retrieve(self, state: dict) -> dict:
        """执行多 Query Retrieval，并把候选和延迟加入 Chain State。

        输入 ``state`` 至少包含 expanded_queries；输出保留全部旧字段，并新增
        ``candidate_documents: list[Document]`` 与 ``retrieval_latency_ms: float``。
        这是普通 Python 方法，稍后用 RunnableLambda 接入 LCEL。
        """
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
        """对统一候选池执行一次 Rerank，并加入 ``retrieved_documents``。

        使用单条 ``retrieval_query`` 与所有候选比较，不对每条 Expansion Query 分别
        Rerank。输出 Document Metadata 继续携带 Citation 和 Fusion 信息。
        """
        documents = rerank_documents(
            state["retrieval_query"],
            state["candidate_documents"],
            settings=self.settings,
            compressor=self._reranker,
        )
        return {**state, "retrieved_documents": documents}

    def _guard(self, state: dict) -> dict:
        """执行 Answer Guard，同时构造最终 Context。

        输出新增 ``context: str``、``refused: bool``、``refusal_reason: str``。后续
        ``RunnableBranch`` 根据 refused 选择固定拒答或 Gemini Answer Generation。
        """
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
        """把内部 Chain State 转换为稳定的 API/Evaluation 结果。

        返回字段包括 Original/Retrieval/Expansion Query、Answer、结构化拒答、Metadata
        Citation、序列化 Documents、Context 和 Retrieval latency。

        ``build_citations`` 与 ``serialize_document`` 都读取同一组 retrieved_documents，
        保证 Answer Context、Debug 文档和 Citation 的来源一致。
        """
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
        """把不同模型返回类型规范化为 ``AnswerOutput`` 并写回 State。

        Structured ChatModel 通常返回 AnswerOutput；某些 Fake/兼容实现可能返回 dict 或
        其他值，因此分别执行直接使用、Pydantic 校验或字符串 fallback。

        如果模型将 refused 设为 True，使用模型给出的 refusal_reason；正常回答时保留
        确定性 Guard 的原因，便于 Debug。
        """
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
        """构建 V3 完整的 LCEL Runnable Pipeline。

        返回：
            Runnable: 输入单个 Question 字符串，输出 ``_finalize`` 结果字典。

        Query Stage：
            ``RunnableParallel`` 把同一字符串送到两路：
            - ``original_question=RunnablePassthrough()``：原样保留用户问题；
            - ``query_processing=processor.as_runnable()``：返回 Retrieval Query 字典。
            Parallel 输出 dict，再交给 ``_merge_query_state`` 扁平化。

        Answer Stage：
            ``answer_inputs`` 从整个 state 中并行抽取 original_question 和 context，得到
            ``{"question": str, "context": str}``，正好匹配 ANSWER_PROMPT 变量。
            Prompt 输出 ChatPromptValue/messages；ChatModel 接收消息并返回 AnswerOutput。
            不支持 Structured Output 的测试模型才走 StrOutputParser fallback。

        Branch：
            Guard 拒答时不调用 Gemini，直接添加固定 answer；否则生成 answer_result，
            规范化后 Finalize。

        LCEL 语法：
            ``A | B`` 会创建顺序 Runnable（概念上是 RunnableSequence），A 的完整输出是
            B 的输入。``lambda`` 是匿名小函数，本方法只用它做简单字段抽取/字符串清理，
            没有把整套旧 RAG 隐藏在单个 Lambda 中。
        """
        processor = self._prepare_query_processor()
        # Phase 9 输入先被规范化为 dict：Answer 保留 original，检索使用 contextual query。
        query_stage = RunnableParallel(
            original_question=RunnableLambda(
                lambda state: state["original_question"]
            ),
            query_processing=(
                RunnableLambda(lambda state: state["retrieval_question"])
                | processor.as_runnable()
            ),
        ) | RunnableLambda(self._merge_query_state).with_config(
            run_name="merge_query_state"
        )

        # 每个 RunnableLambda 都接收完整 state，只抽取 Prompt 需要的一个字段。
        answer_inputs = RunnableParallel(
            question=RunnableLambda(lambda state: state["original_question"]),
            context=RunnableLambda(lambda state: state["context"]),
        )
        model = self._model()
        # 真实 ChatGoogleGenerativeAI 支持 Structured Output；Fake Model 可能不支持。
        if hasattr(model, "with_structured_output"):
            answer_model = model.with_structured_output(AnswerOutput)
            # LCEL：dict -> PromptValue -> AnswerOutput。
            answer_chain = (
                answer_inputs | ANSWER_PROMPT | answer_model
            ).with_config(run_name="answer_generation")
        else:
            # Fallback：dict -> PromptValue -> AIMessage -> str -> AnswerOutput。
            answer_chain = (
                answer_inputs
                | ANSWER_PROMPT
                | model
                | StrOutputParser()
                | RunnableLambda(
                    lambda text: AnswerOutput(answer=text, refused=False)
                )
            ).with_config(run_name="answer_generation_fallback")

        # ``assign`` 保留原 state，并将 answer_chain 的结果放进 answer_result。
        success = (
            RunnablePassthrough.assign(answer_result=answer_chain)
            | RunnableLambda(self._apply_answer_result)
            | RunnableLambda(self._finalize)
        )
        # lambda 参数 ``_`` 表示输入存在但该表达式不需要使用它。
        refusal = (
            RunnablePassthrough.assign(
                answer=lambda _: self.settings.refusal_message
            )
            | RunnableLambda(self._finalize)
        )

        # 多个 ``|`` 组成顺序执行图；最后的 Branch 只会选择一条分支运行。
        return (
            RunnableLambda(self._normalize_chain_input)
            | query_stage
            | RunnableLambda(self._retrieve).with_config(run_name="retrieval")
            | RunnableLambda(self._rerank).with_config(run_name="reranker")
            | RunnableLambda(self._guard).with_config(run_name="answer_guard")
            | RunnableBranch((lambda state: state["refused"], refusal), success)
        ).with_config(run_name="native_rag_v3")

    @property
    def chain(self) -> Runnable:
        """延迟构建并缓存完整 Chain。

        ``@property`` 是装饰器，让调用方写 ``service.chain`` 而不是 ``service.chain()``。
        第一次访问才创建 ChatModel 和 LCEL 图；上传文档刷新 Retriever 后缓存会被清空。
        """
        if self._chain is None:
            self._chain = self.build_rag_chain()
        return self._chain

    def ask_rag(
        self, question: str, *, retrieval_question: str | None = None
    ) -> dict:
        """同步执行完整 RAG Chain。

        输入 ``str``，通过 ``chain.invoke`` 返回完整 ``dict``。FastAPI `/chat` 和 CLI
        ``ask`` 当前都调用本方法，因此会等待整个结果完成后一次性返回。
        """
        value: str | dict = question
        if retrieval_question is not None:
            value = {
                "original_question": question,
                "retrieval_question": retrieval_question,
            }
        return self.chain.invoke(value)

    async def aask_rag(self, question: str) -> dict:
        """异步执行完整 RAG Chain，并返回完整结果。

        ``async def`` 定义协程；调用者必须 ``await``。``await self.chain.ainvoke`` 在等待
        I/O 时允许事件循环处理其他任务，但它仍不是逐 token streaming。
        """
        return await self.chain.ainvoke(question)

    def batch_ask(self, questions: list[str]) -> list[dict]:
        """通过同一条 Chain 批量处理多个问题。

        ``batch`` 接收 ``list[str]``，返回顺序对应的 ``list[dict]``。是否并行以及并发度
        取决于 Runnable 与运行配置；真实 Gemini 批量调用仍需遵守 API 限流。
        """
        return self.chain.batch(questions)

    def retrieve_only(
        self,
        question: str,
        *,
        top_k: int = 10,
        retrieval_question: str | None = None,
    ) -> dict:
        """只运行 Query Processing、Retrieval 和 Reranker，不生成 Answer。

        参数：
            question (str): Original Question。
            top_k (int): Retrieval-only 最终最多返回多少条文档。

        返回：
            dict: Query 调试字段、序列化 Documents 和 latency_ms。

        为什么单独实现：
            Retrieval Evaluation 必须隔离 Answer Generation，才能判断正确 source/page 是否
            被召回，也避免不必要的 Gemini 成本。本方法复用同一个 QueryProcessor、
            Retrieval Engine 和 Reranker，没有另写一套检索算法。

        注意：
            这里直接调用组件而不是完整 Answer LCEL，因为评估需要在生成前停止。若请求的
            top_k 与缓存 compressor 的 top_n 不同，会复用同一模型创建轻量 compressor。
        """
        processor = self._prepare_query_processor()
        query_state = processor.process((retrieval_question or question).strip())
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
    """返回进程内共享的 V3 Service。

    ``@lru_cache(maxsize=1)`` 使 API/CLI 复用已加载的 Embedding、Retriever、Reranker 和
    Chain，避免每次请求重新初始化。测试可调用 ``get_service.cache_clear()`` 清缓存。
    """
    return NativeRAGService()


def build_rag_chain() -> Runnable:
    """模块级便利入口：返回共享 Service 已构建的完整 Runnable。"""
    return get_service().chain


def ask_rag(question: str) -> dict:
    """模块级便利入口：使用共享 Service 同步回答一个问题。"""
    return get_service().ask_rag(question)

