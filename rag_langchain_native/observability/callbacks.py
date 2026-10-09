"""LangChain Callback 到 OpenTelemetry Span/Token Metrics 的桥接。

Callback 比 print 更合适，因为 LCEL、Retriever、ChatModel、OutputParser 都通过统一生命周期
通知 start/end/error，并携带 run_id/parent_run_id。这里不读取 inputs/outputs 内容，只记录
组件名称、数量、耗时和模型返回的 Usage Metadata。
"""

from __future__ import annotations

import threading
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, set_span_in_context

from .failure import classify_exception


def _usage_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return None


def extract_token_usage(response: Any) -> dict[str, Any]:
    """从 LangChain LLMResult/AIMessage 的真实字段提取 Usage。

    优先 ``AIMessage.usage_metadata``，再尝试 ``response_metadata`` 与 ``llm_output``。
    没有真实字段时返回 ``available=False``，绝不使用字符数伪造 Token。
    """
    candidates: list[dict[str, Any]] = []
    generations = getattr(response, "generations", None) or []
    for group in generations:
        for generation in group if isinstance(group, list) else [group]:
            message = getattr(generation, "message", None)
            if message is None:
                continue
            usage = _usage_dict(getattr(message, "usage_metadata", None))
            if usage:
                candidates.append(usage)
                continue
            metadata = getattr(message, "response_metadata", {}) or {}
            for key in ("usage_metadata", "usage", "token_usage"):
                usage = _usage_dict(metadata.get(key))
                if usage:
                    candidates.append(usage)
                    break
    if not candidates:
        llm_output = getattr(response, "llm_output", None) or {}
        for key in ("token_usage", "usage_metadata", "usage"):
            usage = _usage_dict(llm_output.get(key))
            if usage:
                candidates.append(usage)
                break
    if not candidates:
        return {
            "available": False,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    def number(item: dict[str, Any], *keys: str) -> int | None:
        for key in keys:
            if item.get(key) is not None:
                try:
                    return int(item[key])
                except (TypeError, ValueError):
                    return None
        return None

    input_total = output_total = total = 0
    for item in candidates:
        input_value = number(item, "input_tokens", "prompt_tokens", "prompt_token_count")
        output_value = number(item, "output_tokens", "completion_tokens", "candidates_token_count")
        total_value = number(item, "total_tokens", "total_token_count")
        if input_value is None or output_value is None:
            return {
                "available": False,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            }
        input_total += input_value
        output_total += output_value
        total += total_value if total_value is not None else input_value + output_value
    return {
        "available": True,
        "input_tokens": input_total,
        "output_tokens": output_total,
        "total_tokens": total,
    }


class ObservabilityCallback(BaseCallbackHandler):
    """为实际 LangChain Run 创建 Span，并采集真实 Usage。"""

    def __init__(self, manager) -> None:
        self.manager = manager
        self._spans: dict[str, Any] = {}
        self._started: dict[str, float] = {}
        self._names: dict[str, str] = {}
        self._models: dict[str, str | None] = {}
        self._lock = threading.Lock()

    def _start(
        self,
        kind: str,
        run_id,
        parent_run_id=None,
        name: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if not self.manager.settings.enabled:
            return
        key = str(run_id)
        with self._lock:
            if key in self._spans:
                return
        parent = self._spans.get(str(parent_run_id)) if parent_run_id else None
        context = set_span_in_context(parent) if parent else None
        operation = name or kind
        request_id, _, _ = self.manager.ids()
        span = self.manager.tracer.start_span(
            f"langchain.{kind}.{operation}",
            context=context,
            attributes={
                "langchain.kind": kind,
                "langchain.run_name": operation,
                "request.id": request_id or "",
                **(attributes or {}),
            },
        )
        with self._lock:
            self._spans[key] = span
            self._started[key] = time.perf_counter()
            self._names[key] = operation

    def _end(
        self,
        run_id,
        *,
        error: BaseException | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> tuple[str | None, float | None]:
        key = str(run_id)
        with self._lock:
            span = self._spans.pop(key, None)
            started = self._started.pop(key, None)
            name = self._names.pop(key, None)
        if span is None:
            return name, None
        duration = (time.perf_counter() - started) * 1000 if started else None
        if duration is not None:
            span.set_attribute("duration_ms", duration)
        for attr, value in (attributes or {}).items():
            if value is not None:
                span.set_attribute(attr, value)
        if error:
            failure = classify_exception(error, name or "langchain").value
            span.set_status(Status(StatusCode.ERROR, failure))
            span.set_attribute("error.type", failure)
        span.end()
        return name, duration

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, name=None, **kwargs):
        del inputs, tags, metadata, kwargs
        resolved = name or (serialized or {}).get("name") or "chain"
        self._start("chain", run_id, parent_run_id, resolved)

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        del outputs, parent_run_id, kwargs
        self._end(run_id)

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        del parent_run_id, kwargs
        self._end(run_id, error=error)

    def on_retriever_start(self, serialized, query, *, run_id, parent_run_id=None, tags=None, metadata=None, name=None, **kwargs):
        del query, tags, metadata, kwargs
        resolved = name or (serialized or {}).get("name") or "retriever"
        self._start("retriever", run_id, parent_run_id, resolved)

    def on_retriever_end(self, documents, *, run_id, parent_run_id=None, **kwargs):
        del parent_run_id, kwargs
        self._end(run_id, attributes={"result_count": len(documents or [])})

    def on_retriever_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        del parent_run_id, kwargs
        self._end(run_id, error=error)

    def _llm_start(self, serialized, run_id, parent_run_id, name, invocation_params):
        model = (invocation_params or {}).get("model") or (serialized or {}).get("name")
        with self._lock:
            self._models[str(run_id)] = str(model) if model else None
        self._start(
            "llm",
            run_id,
            parent_run_id,
            name or "generate",
            {"model.name": str(model) if model else "unknown"},
        )

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, name=None, **kwargs):
        del messages, tags, metadata
        self._llm_start(serialized, run_id, parent_run_id, name, kwargs.get("invocation_params"))

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, tags=None, metadata=None, name=None, **kwargs):
        del prompts, tags, metadata
        self._llm_start(serialized, run_id, parent_run_id, name, kwargs.get("invocation_params"))

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        del parent_run_id
        usage = extract_token_usage(response)
        with self._lock:
            started_model = self._models.pop(str(run_id), None)
        name, duration = self._end(
            run_id,
            attributes={
                "usage.available": usage["available"],
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
            },
        )
        model = (
            (getattr(response, "llm_output", None) or {}).get("model_name")
            or started_model
        )
        self.manager.add_usage(
            operation=name or "llm.generate",
            model_name=model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            available=usage["available"],
            mocked=bool(kwargs.get("mocked_usage", False)),
        )
        if duration is not None:
            self.manager.metric("llm.duration", duration, kind="histogram", attributes={"operation": name or "generate"})

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        del parent_run_id, kwargs
        with self._lock:
            self._models.pop(str(run_id), None)
        name, _ = self._end(run_id, error=error)
        self.manager.metric("llm.error.count", attributes={"operation": name or "generate", "error_type": classify_exception(error, "llm").value})

    def on_retry(self, retry_state, *, run_id, parent_run_id=None, **kwargs):
        del retry_state, run_id, parent_run_id, kwargs
        self.manager.metric("llm.retry.count")
        self.manager.log("WARNING", operation="llm", event="retry", status="retrying")

