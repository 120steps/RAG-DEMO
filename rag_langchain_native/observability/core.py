"""Phase 10 Tracing、JSON Logging、Metrics 与 Request Summary 的统一实现。

ContextVar 保存当前请求上下文，适用于同步、async/await 与正确传播 Context 的线程任务，
避免使用非线程安全全局变量。业务代码通过 ``request_scope`` 与 ``stage`` 埋点；关闭功能
时仍返回 no-op 上下文，不改变异常、返回值或检索算法。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator
from functools import wraps

from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExportResult,
    SpanExporter,
)
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Status, StatusCode

from .config import DEFAULT_OBSERVABILITY_SETTINGS, ObservabilitySettings
from .cost import PricingCatalog
from .failure import classify_exception, safe_error_summary


SENSITIVE_KEYS = (
    "password", "passwd", "api_key", "apikey", "authorization", "cookie",
    "secret", "jwt", "access_token", "refresh_token", "session_token",
    "prompt", "context", "answer", "document",
    "question", "content",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def tenant_fingerprint(tenant_id: str | None) -> str | None:
    """对租户内部标识做不可逆短摘要；它只用于 Trace 归属，不用于授权。"""
    if not tenant_id:
        return None
    return hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:16]


def redact(value: Any, key: str = "") -> Any:
    """递归清理日志字段；默认不允许业务正文和凭证进入存储。"""
    lowered = key.lower()
    # document_count / question_count 是安全聚合值，不应因字段名含正文类别而被误清理。
    aggregate = lowered.endswith(("_count", "_tokens", "_ms"))
    if not aggregate and any(item in lowered for item in SENSITIVE_KEYS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        if value.lower().startswith("bearer "):
            return "[REDACTED]"
        return value[:512]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:512]


class JsonlWriter:
    """线程安全 JSONL Writer，按大小轮转并清理过期文件。"""

    def __init__(self, path: Path, settings: ObservabilitySettings) -> None:
        self.path = path
        self.settings = settings
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.cleanup()

    def cleanup(self) -> None:
        cutoff = datetime.now(UTC) - timedelta(days=self.settings.retention_days)
        for candidate in self.path.parent.glob(f"{self.path.name}.*"):
            try:
                modified = datetime.fromtimestamp(candidate.stat().st_mtime, UTC)
                if modified < cutoff:
                    candidate.unlink()
            except OSError:
                continue

    def _rotate(self) -> None:
        if not self.path.exists() or self.path.stat().st_size < self.settings.max_file_bytes:
            return
        oldest = self.path.with_name(f"{self.path.name}.{self.settings.backup_count}")
        if oldest.exists():
            oldest.unlink()
        for index in range(self.settings.backup_count - 1, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{index}")
            if source.exists():
                source.replace(self.path.with_name(f"{self.path.name}.{index + 1}"))
        self.path.replace(self.path.with_name(f"{self.path.name}.1"))

    def write(self, value: dict[str, Any]) -> None:
        with self._lock:
            self._rotate()
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(redact(value), ensure_ascii=False) + "\n")


class LocalJsonlSpanExporter(SpanExporter):
    """保留父子关系的本地 OpenTelemetry Span Exporter。"""

    def __init__(self, writer: JsonlWriter) -> None:
        self.writer = writer

    def export(self, spans) -> SpanExportResult:
        for span in spans:
            context = span.context
            parent = span.parent
            self.writer.write(
                {
                    "timestamp": datetime.fromtimestamp(span.start_time / 1e9, UTC).isoformat(),
                    "trace_id": f"{context.trace_id:032x}",
                    "span_id": f"{context.span_id:016x}",
                    "parent_span_id": f"{parent.span_id:016x}" if parent else None,
                    "name": span.name,
                    "start_time_unix_ns": span.start_time,
                    "end_time_unix_ns": span.end_time,
                    "duration_ms": (span.end_time - span.start_time) / 1_000_000,
                    "status": span.status.status_code.name,
                    "attributes": dict(span.attributes or {}),
                    "events": [
                        {"name": event.name, "attributes": dict(event.attributes or {})}
                        for event in span.events
                    ],
                }
            )
        return SpanExportResult.SUCCESS


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "payload", {})
        return json.dumps(
            redact(
                {
                    "timestamp": utc_now(),
                    "level": record.levelname,
                    "service": "rag-langchain-native",
                    **payload,
                }
            ),
            ensure_ascii=False,
        )


@dataclass
class RequestState:
    request_id: str
    operation: str
    started_at: float
    timestamp: str
    tenant_scope: str | None = None
    trace_id: str | None = None
    status: str = "running"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    token_usage_available: bool = False
    mocked_usage: bool = False
    model_names: set[str] = field(default_factory=set)
    estimated_cost: float = 0.0
    cost_status: str = "unknown_pricing"
    currency: str | None = None
    pricing_version: str | None = None
    fallback_count: int = 0
    error_type: str | None = None


_CURRENT_REQUEST: ContextVar[RequestState | None] = ContextVar(
    "v3_observability_request", default=None
)


class ObservabilityManager:
    """一个进程内的可观测性门面。

    Metrics 不使用 request/user/document 等高基数 Label。详细关联信息只进入访问受控的
    request/trace JSONL。OpenTelemetry SDK 与本地聚合同时记录，便于以后接 OTLP 后端。
    """

    def __init__(self, settings: ObservabilitySettings) -> None:
        settings.validate()
        self.settings = settings
        self.pricing = PricingCatalog(settings.pricing_path)
        self.tracer_provider: TracerProvider | None = None
        self.meter_provider: MeterProvider | None = None
        self.tracer = trace.get_tracer(settings.service_name)
        self._counters: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}
        self._configure_logging()
        if settings.enabled:
            settings.ensure_directory()
            self.trace_writer = JsonlWriter(settings.runtime_dir / "traces.jsonl", settings)
            self.request_writer = JsonlWriter(settings.runtime_dir / "requests.jsonl", settings)
            self.metric_writer = JsonlWriter(settings.runtime_dir / "metrics.jsonl", settings)
            resource = Resource.create({"service.name": settings.service_name})
            self.tracer_provider = TracerProvider(
                resource=resource,
                sampler=ParentBased(TraceIdRatioBased(settings.sample_rate)),
            )
            if settings.exporter == "local":
                self.tracer_provider.add_span_processor(
                    SimpleSpanProcessor(LocalJsonlSpanExporter(self.trace_writer))
                )
            elif settings.exporter == "otlp":
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                self.tracer_provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint))
                )
            self.tracer = self.tracer_provider.get_tracer(settings.service_name)
            self.meter_provider = MeterProvider(resource=resource)
            self.meter = self.meter_provider.get_meter(settings.service_name)
        else:
            self.trace_writer = self.request_writer = self.metric_writer = None

    def _configure_logging(self) -> None:
        self.logger = logging.getLogger(f"{self.settings.service_name}.observability.{id(self)}")
        self.logger.setLevel(getattr(logging, self.settings.log_level, logging.INFO))
        self.logger.propagate = False
        self.logger.handlers.clear()
        formatter = JsonLogFormatter()
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        if self.settings.enabled:
            self.settings.runtime_dir.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                self.settings.runtime_dir / "application.jsonl",
                maxBytes=self.settings.max_file_bytes,
                backupCount=self.settings.backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    @property
    def current(self) -> RequestState | None:
        return _CURRENT_REQUEST.get()

    def ids(self) -> tuple[str | None, str | None, str | None]:
        state = self.current
        span = trace.get_current_span().get_span_context()
        trace_id = f"{span.trace_id:032x}" if span.is_valid else (state.trace_id if state else None)
        span_id = f"{span.span_id:016x}" if span.is_valid else None
        return (state.request_id if state else None, trace_id, span_id)

    def log(
        self,
        level: str,
        *,
        operation: str,
        event: str,
        status: str,
        duration_ms: float | None = None,
        error_type: str | None = None,
        **fields: Any,
    ) -> None:
        if not self.settings.enabled:
            return
        request_id, trace_id, _ = self.ids()
        payload = {
            "operation": operation,
            "request_id": request_id,
            "trace_id": trace_id,
            "event": event,
            "status": status,
            "duration_ms": duration_ms,
            "error_type": error_type,
            **fields,
        }
        getattr(self.logger, level.lower(), self.logger.info)("", extra={"payload": payload})

    def _instrument(self, name: str, kind: str):
        target = self._counters if kind == "counter" else self._histograms
        if name not in target and self.settings.enabled:
            target[name] = (
                self.meter.create_counter(name)
                if kind == "counter"
                else self.meter.create_histogram(name, unit="ms")
            )
        return target.get(name)

    def metric(
        self,
        name: str,
        value: float = 1,
        *,
        kind: str = "counter",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if not self.settings.enabled:
            return
        # 调用方只能传低基数技术维度；这里再次丢弃常见高基数名称。
        safe = {
            key: val
            for key, val in (attributes or {}).items()
            if key not in {"request_id", "trace_id", "user_id", "conversation_id", "document_id", "tenant_id"}
        }
        instrument = self._instrument(name, kind)
        if instrument:
            (instrument.add(value, safe) if kind == "counter" else instrument.record(value, safe))
        self.metric_writer.write(
            {"timestamp": utc_now(), "name": name, "value": value, "kind": kind, "attributes": safe}
        )

    @contextmanager
    def request_scope(
        self,
        operation: str,
        *,
        request_id: str | None = None,
        tenant_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[RequestState]:
        """创建根 Trace；若 API Middleware 已创建，则复用当前请求。"""
        existing = self.current
        if existing is not None:
            yield existing
            return
        state = RequestState(
            request_id=request_id or uuid.uuid4().hex,
            operation=operation,
            started_at=time.perf_counter(),
            timestamp=utc_now(),
            tenant_scope=tenant_fingerprint(tenant_id),
        )
        token = _CURRENT_REQUEST.set(state)
        error: BaseException | None = None
        cm = self.tracer.start_as_current_span(
            "rag.request",
            attributes={
                "request.id": state.request_id,
                "operation": operation,
                **(attributes or {}),
            },
        ) if self.settings.enabled else None
        span = cm.__enter__() if cm else None
        if span:
            context = span.get_span_context()
            state.trace_id = f"{context.trace_id:032x}" if context.is_valid else None
        self.metric("api.request.count", attributes={"operation": operation})
        self.log("INFO", operation=operation, event="request.start", status="started")
        try:
            yield state
            state.status = "success"
        except Exception as exc:
            error = exc
            state.status = "error"
            state.error_type = classify_exception(exc, operation).value
            if span:
                span.set_status(Status(StatusCode.ERROR, state.error_type))
                span.set_attribute("error.type", state.error_type)
            raise
        finally:
            duration = (time.perf_counter() - state.started_at) * 1000
            metric_name = "api.success.count" if state.status == "success" else "api.error.count"
            self.metric(metric_name, attributes={"operation": operation})
            self.metric("api.request.duration", duration, kind="histogram", attributes={"operation": operation, "status": state.status})
            if self.settings.enabled:
                self.request_writer.write(
                    {
                        "timestamp": state.timestamp,
                        "request_id": state.request_id,
                        "trace_id": state.trace_id,
                        "tenant_scope": state.tenant_scope,
                        "operation": operation,
                        "status": state.status,
                        "duration_ms": duration,
                        "input_tokens": state.input_tokens if state.token_usage_available else None,
                        "output_tokens": state.output_tokens if state.token_usage_available else None,
                        "total_tokens": state.total_tokens if state.token_usage_available else None,
                        "usage_status": "mocked" if state.mocked_usage else ("available" if state.token_usage_available else "unavailable"),
                        "models": sorted(state.model_names),
                        "estimated_cost": state.estimated_cost if state.cost_status == "estimated" else None,
                        "cost_status": state.cost_status,
                        "currency": state.currency,
                        "pricing_version": state.pricing_version,
                        "fallback_count": state.fallback_count,
                        "error_type": state.error_type,
                    }
                )
            self.log(
                "ERROR" if error else "INFO",
                operation=operation,
                event="request.end",
                status=state.status,
                duration_ms=duration,
                error_type=state.error_type,
            )
            if cm:
                cm.__exit__(
                    type(error) if error else None,
                    error,
                    error.__traceback__ if error else None,
                )
            _CURRENT_REQUEST.reset(token)

    def bind_tenant(self, tenant_id: str) -> None:
        """认证成功后把租户的不可逆指纹绑定到当前请求，而不保存原始 tenant_id。"""
        state = self.current
        if state is None:
            return
        state.tenant_scope = tenant_fingerprint(tenant_id)
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("tenant.scope", state.tenant_scope or "")

    @contextmanager
    def stage(
        self,
        operation: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """记录实际执行阶段。yield 的字典允许调用方在结束前补充 result_count 等字段。"""
        output: dict[str, Any] = {}
        if not self.settings.enabled:
            yield output
            return
        started = time.perf_counter()
        error: BaseException | None = None
        with self.tracer.start_as_current_span(operation, attributes=attributes or {}) as span:
            try:
                yield output
            except Exception as exc:
                error = exc
                failure = classify_exception(exc, operation).value
                span.set_status(Status(StatusCode.ERROR, failure))
                span.set_attribute("error.type", failure)
                self.metric("failure.count", attributes={"operation": operation, "error_type": failure})
                self.log("ERROR", operation=operation, event="stage.error", status="error", error_type=failure, safe_summary=safe_error_summary(exc))
                raise
            finally:
                duration = (time.perf_counter() - started) * 1000
                span.set_attribute("duration_ms", duration)
                span.set_attribute("success", error is None)
                for key, value in output.items():
                    if value is not None and isinstance(value, (str, bool, int, float)):
                        span.set_attribute(key, value)
                self.metric(f"{operation}.duration", duration, kind="histogram", attributes={"status": "error" if error else "success"})
                self.log("INFO" if not error else "ERROR", operation=operation, event="stage.end", status="error" if error else "success", duration_ms=duration, **output)

    def fallback(self, operation: str, error: BaseException) -> None:
        """记录局部降级；不会把最终成功请求误记为失败。"""
        state = self.current
        if state:
            state.fallback_count += 1
        error_type = classify_exception(error, operation).value
        self.metric("fallback.count", attributes={"operation": operation, "error_type": error_type})
        self.log("WARNING", operation=operation, event="fallback", status="degraded", error_type=error_type, safe_summary=safe_error_summary(error))

    def add_usage(
        self,
        *,
        operation: str,
        model_name: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        available: bool,
        mocked: bool = False,
    ) -> None:
        state = self.current
        if model_name and state:
            state.model_names.add(model_name)
        if state and available and input_tokens is not None and output_tokens is not None:
            state.token_usage_available = True
            state.mocked_usage = state.mocked_usage or mocked
            state.input_tokens += input_tokens
            state.output_tokens += output_tokens
            state.total_tokens += total_tokens if total_tokens is not None else input_tokens + output_tokens
        estimate = self.pricing.estimate(
            model_name, input_tokens, output_tokens,
            usage_available=available, mocked_usage=mocked,
        )
        if state:
            state.cost_status = estimate.status
            if estimate.status == "estimated" and estimate.estimated_cost is not None:
                state.estimated_cost += estimate.estimated_cost
                state.currency = estimate.currency
                state.pricing_version = estimate.pricing_version
        self.metric("llm.request.count", attributes={"operation": operation, "usage": "available" if available else "unavailable"})
        if available and input_tokens is not None:
            self.metric("llm.input_tokens", input_tokens, attributes={"operation": operation})
            self.metric("llm.output_tokens", output_tokens or 0, attributes={"operation": operation})

    def callback(self):
        """为当前请求创建 LangChain Callback Handler。"""
        from .callbacks import ObservabilityCallback

        return ObservabilityCallback(self)

    def langchain_config(self) -> dict[str, Any]:
        return {"callbacks": [self.callback()]} if self.settings.enabled else {}

    def shutdown(self) -> None:
        if self.tracer_provider:
            self.tracer_provider.force_flush()
            self.tracer_provider.shutdown()
        if self.meter_provider:
            self.meter_provider.shutdown()


_MANAGER: ObservabilityManager | None = None
_MANAGER_LOCK = threading.Lock()


def configure_observability(settings: ObservabilitySettings) -> ObservabilityManager:
    """测试/启动时显式替换进程 Manager；不会修改 V1/V2。"""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is not None:
            _MANAGER.shutdown()
        _MANAGER = ObservabilityManager(settings)
        return _MANAGER


def get_observability() -> ObservabilityManager:
    global _MANAGER
    if _MANAGER is None:
        with _MANAGER_LOCK:
            if _MANAGER is None:
                _MANAGER = ObservabilityManager(DEFAULT_OBSERVABILITY_SETTINGS)
    return _MANAGER


def observed_request(operation: str):
    """让非 HTTP 的 Service 入口也拥有根 Trace；嵌套在 API 中时自动复用已有请求。"""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            principal = kwargs.get("principal")
            tenant_id = getattr(principal, "tenant_id", None)
            manager = get_observability()
            with manager.request_scope(operation, tenant_id=tenant_id):
                return function(*args, **kwargs)
        return wrapped
    return decorate

