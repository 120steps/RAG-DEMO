"""LCEL query rewrite and expansion with V3-owned caches."""

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
    queries: list[str] = Field(
        description="Only alternative search queries; do not include answers"
    )


class JsonCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def read(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.exists():
                return {}
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
                return value if isinstance(value, dict) else {}
            except (OSError, json.JSONDecodeError):
                return {}

    def set(self, key: str, value: Any) -> None:
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
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._last_request = 0.0

    def wait(self) -> None:
        with self._lock:
            remaining = self.interval_seconds - (time.monotonic() - self._last_request)
            if remaining > 0:
                time.sleep(remaining)
            self._last_request = time.monotonic()


class QueryProcessor:
    def __init__(
        self,
        model: Runnable | None,
        settings: Settings = DEFAULT_SETTINGS,
    ) -> None:
        self.model = model
        self.settings = settings
        settings.ensure_runtime_dirs()
        self.rewrite_cache = JsonCache(
            settings.cache_dir / "query_rewrite_cache.json"
        )
        self.expansion_cache = JsonCache(
            settings.cache_dir / "query_expansion_cache.json"
        )
        self.throttle = RequestThrottle(settings.gemini_min_interval_seconds)

    def _require_model(self) -> Runnable:
        if self.model is None:
            raise RuntimeError("Gemini model is unavailable")
        return self.model

    def rewrite(self, question: str) -> tuple[str, bool]:
        if not self.settings.rewrite_enabled:
            return question, False
        cached = self.rewrite_cache.read().get(question)
        if isinstance(cached, str) and cached.strip():
            return cached.strip(), False
        try:
            self.throttle.wait()
            chain = REWRITE_PROMPT | self._require_model() | StrOutputParser()
            rewritten = chain.invoke({"question": question}).strip()
            if not rewritten:
                raise ValueError("Gemini returned an empty rewrite")
            self.rewrite_cache.set(question, rewritten)
            return rewritten, False
        except Exception:
            return question, True

    def expand(self, query: str) -> tuple[list[str], bool]:
        if not self.settings.expansion_enabled:
            return [query], False
        cached = self.expansion_cache.read().get(query)
        if isinstance(cached, list):
            cached_queries = [str(item).strip() for item in cached if str(item).strip()]
            if cached_queries:
                return list(dict.fromkeys([query, *cached_queries])), False
        try:
            self.throttle.wait()
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
            self.expansion_cache.set(query, queries)
            return queries, False
        except Exception:
            return [query], True

    def process(self, question: str) -> dict:
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
        return RunnableLambda(self.process).with_config(run_name="query_processing")

