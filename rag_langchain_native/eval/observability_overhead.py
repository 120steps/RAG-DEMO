"""离线测量 Observability 的框架开销；工作负载不调用模型、Embedding 或数据库。"""

from __future__ import annotations

import json
import statistics
import sys
import time
import uuid
from pathlib import Path

from rag_langchain_native.observability.config import ObservabilitySettings
from rag_langchain_native.observability.core import ObservabilityManager
from rag_langchain_native.observability.report import LocalReportStore


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "eval" / "results" / "phase10_observability_overhead.json"
RUNTIME = ROOT / "runtime" / "observability_benchmark"
SAMPLES = 200


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * ratio + 0.9999) - 1)]


def run(enabled: bool) -> dict:
    # 每次使用新的受控子目录，避免历史 Trace 被误计入本轮完整性统计。
    path = RUNTIME / ("on" if enabled else "off") / uuid.uuid4().hex
    settings = ObservabilitySettings(
        enabled=enabled,
        exporter="local",
        sample_rate=1.0,
        log_level="CRITICAL",
        capture_content=False,
        retention_days=1,
        max_file_bytes=10_000_000,
        backup_count=1,
        runtime_dir=path,
        pricing_path=path / "pricing.json",
    )
    manager = ObservabilityManager(settings)
    latencies = []
    errors = 0
    for index in range(SAMPLES):
        started = time.perf_counter()
        try:
            with manager.request_scope("benchmark.mock", request_id=f"bench-{index}"):
                with manager.stage("retrieval") as data:
                    # 固定的轻量 CPU 工作，避免把外部模型抖动误算成监控开销。
                    sum(value * value for value in range(100))
                    data["result_count"] = 10
                with manager.stage("reranker") as data:
                    sum(value * value for value in range(50))
                    data["result_count"] = 3
        except Exception:
            errors += 1
        latencies.append((time.perf_counter() - started) * 1000)
    if manager.tracer_provider:
        manager.tracer_provider.force_flush()
    span_count = len(LocalReportStore(settings).spans()) if enabled else 0
    manager.shutdown()
    return {
        "enabled": enabled,
        "samples": SAMPLES,
        "average_latency_ms": statistics.fmean(latencies),
        "p95_latency_ms": percentile(latencies, 0.95),
        "error_count": errors,
        "span_count": span_count,
        "expected_span_count": SAMPLES * 3 if enabled else 0,
        "trace_complete": span_count == SAMPLES * 3 if enabled else True,
    }


def main() -> None:
    disabled = run(False)
    enabled = run(True)
    overhead = enabled["average_latency_ms"] - disabled["average_latency_ms"]
    result = {
        "workload": "local mock: request + retrieval span + reranker span",
        "note": "微型工作负载会放大相对开销，不能代表真实 Gemini 端到端性能。",
        "disabled": disabled,
        "enabled": enabled,
        "average_absolute_overhead_ms": overhead,
        "average_relative_overhead_percent": (
            overhead / disabled["average_latency_ms"] * 100
            if disabled["average_latency_ms"] else None
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
