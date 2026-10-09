"""Phase 10 可观测性离线测试：不调用 Gemini，不读取真实业务文档。"""

from __future__ import annotations

import json
import uuid

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from rag_langchain_native.observability.callbacks import extract_token_usage
from rag_langchain_native.observability.config import ObservabilitySettings
from rag_langchain_native.observability.core import ObservabilityManager, redact
from rag_langchain_native.observability.core import tenant_fingerprint
from rag_langchain_native.observability.cost import PricingCatalog
from rag_langchain_native.observability.failure import FailureType, classify_exception
from rag_langchain_native.observability.report import LocalReportStore
from rag_langchain_native.security import AuthenticationError, AuthorizationError


def settings(tmp_path, *, enabled=True) -> ObservabilitySettings:
    runtime = tmp_path / "observability"
    return ObservabilitySettings(
        enabled=enabled,
        exporter="local",
        sample_rate=1.0,
        log_level="CRITICAL",
        capture_content=False,
        retention_days=1,
        max_file_bytes=1024 * 1024,
        backup_count=2,
        runtime_dir=runtime,
        pricing_path=runtime / "pricing.json",
    )


def test_trace_parent_child_request_summary_and_report(tmp_path):
    manager = ObservabilityManager(settings(tmp_path))
    with manager.request_scope("rag.test", request_id="request-1", tenant_id="tenant-a"):
        with manager.stage("retrieval") as result:
            result["result_count"] = 3
        with manager.stage("reranker") as result:
            result["input_count"] = 3
            result["result_count"] = 2
    manager.tracer_provider.force_flush()

    store = LocalReportStore(manager.settings)
    detail = store.trace_detail(request_id="request-1")
    assert detail is not None
    def flatten(nodes):
        return [node for item in nodes for node in [item, *flatten(item.get("children", []))]]

    flat = flatten(detail["spans"])
    names = {span["name"] for span in flat}
    assert {"rag.request", "retrieval", "reranker"} <= names
    root = next(span for span in flat if span["name"] == "rag.request")
    assert {span["name"] for span in root["children"]} == {"retrieval", "reranker"}
    assert store.summary(minutes=60)["request_count"] == 1
    manager.shutdown()


def test_trace_detail_enforces_tenant_scope(tmp_path):
    manager = ObservabilityManager(settings(tmp_path))
    with manager.request_scope("rag.test", request_id="tenant-request", tenant_id="tenant-a") as state:
        trace_id = state.trace_id
    manager.tracer_provider.force_flush()
    store = LocalReportStore(manager.settings)
    assert store.trace_detail(trace_id=trace_id, tenant_scope=tenant_fingerprint("tenant-a"))
    assert store.trace_detail(trace_id=trace_id, tenant_scope=tenant_fingerprint("tenant-b")) is None
    manager.shutdown()


def test_observability_off_writes_nothing(tmp_path):
    manager = ObservabilityManager(settings(tmp_path, enabled=False))
    with manager.request_scope("rag.off"):
        with manager.stage("retrieval"):
            pass
    assert not manager.settings.runtime_dir.exists()


def test_token_usage_real_metadata_and_unavailable():
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
    )
    response = LLMResult(generations=[[ChatGeneration(message=message)]])
    usage = extract_token_usage(response)
    assert usage == {
        "available": True,
        "input_tokens": 11,
        "output_tokens": 4,
        "total_tokens": 15,
    }
    assert extract_token_usage(LLMResult(generations=[]))["available"] is False


def test_cost_requires_explicit_verified_pricing(tmp_path):
    path = tmp_path / "pricing.json"
    catalog = PricingCatalog(path)
    assert catalog.estimate("model-x", 10, 5, usage_available=True).status == "unknown_pricing"
    path.write_text(json.dumps({
        "currency": "USD",
        "pricing_version": "test-only",
        "models": {"model-x": {"input_per_million": 1.0, "output_per_million": 2.0}},
    }), encoding="utf-8")
    estimate = catalog.estimate("model-x", 1_000_000, 500_000, usage_available=True)
    assert estimate.status == "estimated"
    assert estimate.estimated_cost == 2.0
    assert catalog.estimate("model-x", 10, 5, usage_available=True, mocked_usage=True).status == "mocked_usage"


def test_redaction_keeps_usage_but_removes_credentials_and_content():
    value = redact({
        "password": "p-secret",
        "authorization": "Bearer abc",
        "input_tokens": 12,
        "context": "confidential document",
        "document_count": 5,
        "safe": "Bearer hidden",
    })
    assert value["password"] == "[REDACTED]"
    assert value["authorization"] == "[REDACTED]"
    assert value["context"] == "[REDACTED]"
    assert value["safe"] == "[REDACTED]"
    assert value["input_tokens"] == 12
    assert value["document_count"] == 5


def test_structured_log_is_redacted(tmp_path):
    configured = settings(tmp_path)
    manager = ObservabilityManager(ObservabilitySettings(**{
        **configured.__dict__, "log_level": "DEBUG"
    }))
    with manager.request_scope("rag.log"):
        manager.log(
            "ERROR", operation="auth", event="failed", status="error",
            password="never-write-me", input_tokens=7,
        )
    content = (manager.settings.runtime_dir / "application.jsonl").read_text(encoding="utf-8")
    assert "never-write-me" not in content
    assert "[REDACTED]" in content
    assert '"input_tokens": 7' in content
    manager.shutdown()


def test_failure_classification_and_fallback(tmp_path):
    assert classify_exception(AuthenticationError("bad")) is FailureType.AUTHENTICATION_FAILURE
    assert classify_exception(AuthorizationError("bad")) is FailureType.AUTHORIZATION_FAILURE
    manager = ObservabilityManager(settings(tmp_path))
    with manager.request_scope("rag.fallback") as state:
        manager.fallback("query.rewrite", TimeoutError("private details"))
        assert state.fallback_count == 1
    row = LocalReportStore(manager.settings).recent(1)[0]
    assert row["status"] == "success"
    assert row["fallback_count"] == 1
    manager.shutdown()


def test_callback_creates_llm_span_and_usage(tmp_path):
    manager = ObservabilityManager(settings(tmp_path))
    callback = manager.callback()
    run_id = uuid.uuid4()
    response = LLMResult(generations=[[ChatGeneration(message=AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    ))]])
    with manager.request_scope("rag.callback") as state:
        callback.on_chat_model_start(
            {"name": "fake-chat"}, [[]], run_id=run_id,
            invocation_params={"model": "fake-chat"}, name="answer_generation",
        )
        callback.on_llm_end(response, run_id=run_id)
        assert state.input_tokens == 3
        assert state.output_tokens == 2
    manager.tracer_provider.force_flush()
    spans = LocalReportStore(manager.settings).spans()
    assert any(span["name"] == "langchain.llm.answer_generation" for span in spans)
    manager.shutdown()


def test_fastapi_returns_request_and_trace_ids(tmp_path, monkeypatch):
    """无效登录仍应可关联，但响应和日志都不能包含提交的密码。"""
    from rag_langchain_native import api
    from rag_langchain_native.observability.core import configure_observability

    manager = configure_observability(settings(tmp_path))
    client = TestClient(api.create_app())
    response = client.post("/auth/login", json={
        "tenant_id": "missing", "username": "nobody", "password": "not-a-real-password"
    })
    assert response.status_code == 401
    assert response.headers["x-request-id"]
    assert len(response.headers["x-trace-id"]) == 32
    manager.tracer_provider.force_flush()
    logs = (manager.settings.runtime_dir / "application.jsonl").read_text(encoding="utf-8")
    assert "not-a-real-password" not in logs
