"""V3 的 Query Rewrite / Query Expansion 模块。

文件职责：
    在 Embedding、Vector Search 和 BM25 之前，把 Original Question 转成更适合检索的
    Query；按配置可再生成多个检索表达。这里只生成搜索 Query，不回答用户问题。

Rewrite 与 Expansion 的区别：
    - Rewrite：一条口语或模糊问题 -> 一条更清晰的 Retrieval Query。
    - Expansion：保留该 Retrieval Query，再生成多个不同检索角度，提高 Recall。

输入与输出：
    输入是 ``question: str``；输出字典包含 ``retrieval_query: str``、
    ``expanded_queries: list[str]`` 和失败标记。Original Question 由上游 LCEL 并行保留，
    最终 Answer 仍回答原问题，不把 Rewrite 结果冒充用户问题。

可靠性设计：
    两项功能默认关闭；开启时先读 V3 独立 JSON Cache，再限流调用 Gemini。模型或文件
    失败时回退原 Query，不能让 Query Processing 故障拖垮整个 RAG 请求。

LangChain 组件：
    ``ChatPromptTemplate``、``Runnable``、``RunnableLambda``、``StrOutputParser`` 和
    Structured Output。当前没有使用 ``MultiQueryRetriever``；Expansion Query 的检索与
    RRF 在 ``retrieval.py`` 完成。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from .config import DEFAULT_SETTINGS, Settings


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是企业知识库检索查询改写器。只改写检索表达，不回答问题。"
            "保留原始意图以及所有数字、日期、人名、组织、协议名、产品型号和缩写。"
            "不得补充未知事实，不要解释，只返回一条查询。",
        ),
        ("human", "{question}"),
    ]
)

EXPANSION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是企业知识库检索查询扩展器。生成 {count} 条不同检索角度的查询。"
            "它们必须保持原始意图，不回答问题，不推断事实，不得改写或删除数字、日期、"
            "人名、组织名、产品型号、协议和缩写（如 MACsec、BGP、OSPF、NAC、802.1X）。"
            "避免语义重复。",
        ),
        ("human", "原始检索 Query：{query}"),
    ]
)


class ExpansionOutput(BaseModel):
    """Query Expansion 的结构化模型输出。

    Pydantic ``BaseModel`` 用字段声明约束输出结构。模型必须返回 ``queries: list[str]``，
    下游不需要从一段自由文本中猜测 JSON 格式。
    """

    queries: list[str] = Field(
        description="Only alternative search queries; do not include answers"
    )


class JsonCache:
    """线程安全的 V3 Query Processing JSON Cache。

    Cache 键是输入 Query，值是 Rewrite 字符串或 Expansion 列表。它属于可靠性与成本
    控制逻辑，LangChain 不会自动替项目维护这种文件缓存。
    """

    def __init__(self, path: Path) -> None:
        """保存 Cache 路径，并创建保护同一进程并发访问的 Lock。"""
        self.path = path
        self._lock = threading.Lock()

    def read(self) -> dict[str, Any]:
        """读取整个 Cache；文件不存在、损坏或不是字典时返回空字典。

        返回：
            dict[str, Any]: Cache 内容。

        初学者知识点：
            ``with self._lock`` 是上下文管理器：进入代码块时加锁，离开时即使发生异常
            也会释放锁。``try/except`` 把可恢复的文件或 JSON 错误转换为空 Cache。
        """
        with self._lock:
            if not self.path.exists():
                return {}
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
                return value if isinstance(value, dict) else {}
            except (OSError, json.JSONDecodeError):
                return {}

    def set(self, key: str, value: Any) -> None:
        """以原子替换方式写入一个 Cache 值。

        参数：
            key (str): Original Question 或 Retrieval Query。
            value (Any): Rewrite 字符串或 Expansion 列表。

        执行过程：
            加锁 -> 读取旧字典 -> 更新 key -> UTF-8 写临时文件 -> ``replace`` 正式文件。
            临时文件替换能降低进程在写到一半时留下残缺 JSON 的风险。

        ``ensure_ascii=False`` 保持中文可读；该方法没有返回值。
        """
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                cache = (
                    json.loads(self.path.read_text(encoding="utf-8"))
                    if self.path.exists()
                    else {}
                )
            except (OSError, json.JSONDecodeError):
                cache = {}
            if not isinstance(cache, dict):
                cache = {}
            cache[key] = value
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)


class RequestThrottle:
    """限制同一进程内两次 Query LLM 请求的最短间隔。"""

    def __init__(self, interval_seconds: float) -> None:
        """记录最短秒数、互斥锁和上一次请求时间。"""
        self.interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._last_request = 0.0

    def wait(self) -> None:
        """必要时阻塞等待，避免连续 Gemini 请求触发限流。

        ``time.monotonic()`` 只关注经过时长，不受系统时钟调整影响。这个限流器不能替代
        分布式限流；它只保护当前 Python 进程。
        """
        with self._lock:
            remaining = self.interval_seconds - (time.monotonic() - self._last_request)
            if remaining > 0:
                time.sleep(remaining)
            self._last_request = time.monotonic()


class QueryProcessor:
    """协调 Query Rewrite、Expansion、Cache、限流和失败回退。

    ``model`` 使用抽象的 Runnable 类型，因此生产可传 ChatGoogleGenerativeAI，测试可传
    Fake Runnable。``self`` 表示当前 QueryProcessor 实例，保存共享配置和 Cache 对象。
    """

    def __init__(
        self,
        model: Runnable | None,
        settings: Settings = DEFAULT_SETTINGS,
        cache_namespace: str = "default",
    ) -> None:
        """创建 QueryProcessor 及其 V3 独立 Cache。

        参数：
            model (Runnable | None): 可调用的 ChatModel；两项功能都关闭时可以为 None。
            settings (Settings): 开关、Expansion 数量、Cache 路径和限流间隔。
            cache_namespace (str): Phase 9 的租户、知识库、权限版本作用域。默认值保留
                Phase 8 原缓存格式；企业主链始终传入隔离 Namespace。
        """
        self.model = model
        self.settings = settings
        self.cache_namespace = cache_namespace
        settings.ensure_runtime_dirs()
        self.rewrite_cache = JsonCache(
            settings.cache_dir / "query_rewrite_cache.json"
        )
        self.expansion_cache = JsonCache(
            settings.cache_dir / "query_expansion_cache.json"
        )
        self.throttle = RequestThrottle(settings.gemini_min_interval_seconds)

    def _cache_key(self, value: str) -> str:
        """把租户、知识库、权限版本作用域加入 Cache Key，防止跨边界复用。"""
        # default 保持 Phase 8 已有 Cache 格式；企业请求一定使用非 default 作用域。
        return value if self.cache_namespace == "default" else f"{self.cache_namespace}\n{value}"

    def _require_model(self) -> Runnable:
        """返回可用模型；需要模型但未配置时抛出明确错误。

        该异常会在 ``rewrite`` / ``expand`` 的 try/except 中被捕获并触发回退，不会让
        完整 RAG 失败。
        """
        if self.model is None:
            raise RuntimeError("Gemini model is unavailable")
        return self.model

    def rewrite(self, question: str) -> tuple[str, bool]:
        """把一条 Original Question 改写为一条 Retrieval Query。

        参数：
            question (str): 用户原始问题。

        返回：
            tuple[str, bool]: ``(rewritten_query, rewrite_failed)``。关闭功能或命中 Cache
            时 failed=False；调用失败时返回原问题并标记 True。

        执行过程：
            1. 开关关闭则完全不调用模型。
            2. Cache 有非空字符串则直接返回。
            3. 限流后构造 ``REWRITE_PROMPT | model | StrOutputParser()``。
            4. ``invoke({"question": question})`` 同步执行链并得到字符串。
            5. 成功写 Cache；任何异常回退原问题。

        LCEL 管道解释：
            Prompt 接收字典并输出 ChatPromptValue/messages；ChatModel 接收消息并返回
            AIMessage；StrOutputParser 把 AIMessage 的文本内容转换为 ``str``。这些对象
            都实现 Runnable 协议，所以 ``|`` 被重载为顺序组合，不是 Python 类型联合。
        """
        if not self.settings.rewrite_enabled:
            return question, False
        cache_key = self._cache_key(question)
        cached = self.rewrite_cache.read().get(cache_key)
        if isinstance(cached, str) and cached.strip():
            return cached.strip(), False
        try:
            self.throttle.wait()
            # 这里的 ``|`` 是 LCEL 管道；与 ``Runnable | None`` 类型注解中的 ``|`` 含义不同。
            chain = REWRITE_PROMPT | self._require_model() | StrOutputParser()
            rewritten = chain.invoke({"question": question}).strip()
            if not rewritten:
                raise ValueError("Gemini returned an empty rewrite")
            self.rewrite_cache.set(cache_key, rewritten)
            return rewritten, False
        except Exception:
            return question, True

    def expand(self, query: str) -> tuple[list[str], bool]:
        """生成多条同意图 Retrieval Query，并始终保留输入 Query。

        参数：
            query (str): Rewrite 后的 Query；Rewrite 关闭时就是 Original Question。

        返回：
            tuple[list[str], bool]: ``(queries, expansion_failed)``。列表第一个元素始终是
            输入 query；失败时退化为 ``[query]``。

        执行过程：
            1. 关闭时直接返回单元素列表。
            2. Cache 命中时用 ``dict.fromkeys`` 保序去重。
            3. ``with_structured_output(ExpansionOutput)`` 要求模型返回 Pydantic 对象。
            4. Prompt 与 structured model 通过 LCEL 连接并 ``invoke``。
            5. 清理空字符串、去重、限制为 original + expansion_count。
            6. 成功写 Cache；异常回退原 Query。

        注意：
            多 Query 只用于 Retrieval。它们会在 ``NativeRetrievalEngine.retrieve_queries``
            中批量检索并统一 Fusion，之后只进行一次 Rerank。
        """
        if not self.settings.expansion_enabled:
            return [query], False
        cache_key = self._cache_key(query)
        cached = self.expansion_cache.read().get(cache_key)
        if isinstance(cached, list):
            cached_queries = [str(item).strip() for item in cached if str(item).strip()]
            if cached_queries:
                return list(dict.fromkeys([query, *cached_queries])), False
        try:
            self.throttle.wait()
            # Structured Output 比让模型返回自由格式 JSON 字符串更容易验证字段类型。
            structured_model = self._require_model().with_structured_output(
                ExpansionOutput
            )
            chain = EXPANSION_PROMPT | structured_model
            output = chain.invoke(
                {"query": query, "count": self.settings.expansion_count}
            )
            alternatives = [item.strip() for item in output.queries if item.strip()]
            queries = list(dict.fromkeys([query, *alternatives]))[
                : self.settings.expansion_count + 1
            ]
            self.expansion_cache.set(cache_key, queries)
            return queries, False
        except Exception:
            return [query], True

    def process(self, question: str) -> dict:
        """按 Rewrite -> Expansion 的固定顺序处理一条问题。

        参数：
            question (str): Original Question。

        返回：
            dict: 包含 ``retrieval_query: str``、``expanded_queries: list[str]``、
            ``rewrite_failed: bool``、``expansion_failed: bool``。

        调用关系：
            上游是 V3 Query Stage 和 Retrieval-only 入口；下游是 ``rewrite``、``expand``。
            Original Question 不在这里返回，因为主 LCEL 的 RunnableParallel 另一路会保留。
        """
        original = question.strip()
        retrieval_query, rewrite_failed = self.rewrite(original)
        expanded_queries, expansion_failed = self.expand(retrieval_query)
        return {
            "retrieval_query": retrieval_query,
            "expanded_queries": expanded_queries,
            "rewrite_failed": rewrite_failed,
            "expansion_failed": expansion_failed,
        }

    def as_runnable(self) -> RunnableLambda:
        """把普通 ``process`` 方法适配成可加入 LCEL 的 RunnableLambda。

        返回：
            RunnableLambda: 输入 ``str``，输出 ``process`` 的字典。

        为什么这样做：
            QueryProcessor 内部的 Cache/限流适合普通 Python；主流程又需要统一 Runnable
            编排。只把这一阶段包装为节点，而不是把旧版完整 ``ask_rag`` 包成一个节点，
            能保留阶段边界和 tracing 名称 ``query_processing``。
        """
        return RunnableLambda(self.process).with_config(run_name="query_processing")

