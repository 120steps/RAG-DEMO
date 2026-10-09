"""不依赖 SaaS/Docker 的本地 Trace、Request 与 Metrics 查看工具。

CLI 示例：
    python -m rag_langchain_native.observability.report recent --limit 10
    python -m rag_langchain_native.observability.report trace --request-id <id>
    python -m rag_langchain_native.observability.report summary --minutes 60
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import DEFAULT_OBSERVABILITY_SETTINGS, ObservabilitySettings


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


class LocalReportStore:
    """只读本地 JSONL；API 调用方必须在外层完成 admin/tenant 授权。"""

    def __init__(self, settings: ObservabilitySettings = DEFAULT_OBSERVABILITY_SETTINGS) -> None:
        self.settings = settings

    @staticmethod
    def _read_files(path: Path) -> list[dict[str, Any]]:
        rows = []
        candidates = sorted(path.parent.glob(f"{path.name}*"), key=lambda item: item.stat().st_mtime)
        for candidate in candidates:
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    try:
                        value = json.loads(line)
                        if isinstance(value, dict):
                            rows.append(value)
                    except json.JSONDecodeError:
                        continue
            except OSError:
                continue
        return rows

    def requests(self, tenant_scope: str | None = None) -> list[dict[str, Any]]:
        rows = self._read_files(self.settings.runtime_dir / "requests.jsonl")
        if tenant_scope is not None:
            rows = [row for row in rows if row.get("tenant_scope") == tenant_scope]
        return rows

    def spans(self) -> list[dict[str, Any]]:
        return self._read_files(self.settings.runtime_dir / "traces.jsonl")

    def recent(self, limit: int = 10, tenant_scope: str | None = None) -> list[dict[str, Any]]:
        return list(reversed(self.requests(tenant_scope)[-limit:]))

    def trace_detail(
        self,
        *,
        request_id: str | None = None,
        trace_id: str | None = None,
        tenant_scope: str | None = None,
    ) -> dict[str, Any] | None:
        requests = self.requests(tenant_scope)
        request = next(
            (
                row for row in reversed(requests)
                if (request_id and row.get("request_id") == request_id)
                or (trace_id and row.get("trace_id") == trace_id)
            ),
            None,
        )
        # tenant_scope 存在时，必须先在该租户的 request summary 中找到归属；仅知道 Trace ID
        # 不是授权，不能借此读取另一个租户的 Span。
        if tenant_scope is not None and request is None:
            return None
        selected_trace = trace_id or (request or {}).get("trace_id")
        if not selected_trace:
            return None
        spans = [row for row in self.spans() if row.get("trace_id") == selected_trace]
        by_parent: dict[str | None, list[dict[str, Any]]] = {}
        for span in spans:
            by_parent.setdefault(span.get("parent_span_id"), []).append(span)

        def children(parent_id: str | None) -> list[dict[str, Any]]:
            values = []
            for span in sorted(by_parent.get(parent_id, []), key=lambda item: item.get("start_time_unix_ns", 0)):
                values.append(
                    {
                        "name": span.get("name"),
                        "span_id": span.get("span_id"),
                        "status": span.get("status"),
                        "duration_ms": span.get("duration_ms"),
                        "attributes": span.get("attributes", {}),
                        "children": children(span.get("span_id")),
                    }
                )
            return values

        span_ids = {span.get("span_id") for span in spans}
        roots = [span for span in spans if span.get("parent_span_id") not in span_ids]
        tree = []
        for root in sorted(roots, key=lambda item: item.get("start_time_unix_ns", 0)):
            tree.append(
                {
                    "name": root.get("name"),
                    "span_id": root.get("span_id"),
                    "status": root.get("status"),
                    "duration_ms": root.get("duration_ms"),
                    "attributes": root.get("attributes", {}),
                    "children": children(root.get("span_id")),
                }
            )
        return {"request": request, "trace_id": selected_trace, "spans": tree}

    def summary(
        self,
        minutes: int = 60,
        tenant_scope: str | None = None,
    ) -> dict[str, Any]:
        cutoff = datetime.now(UTC) - timedelta(minutes=minutes)
        rows = []
        for row in self.requests(tenant_scope):
            try:
                if datetime.fromisoformat(row["timestamp"]) >= cutoff:
                    rows.append(row)
            except (KeyError, ValueError, TypeError):
                continue
        latencies = [float(row["duration_ms"]) for row in rows if row.get("duration_ms") is not None]
        error_types = Counter(row.get("error_type") for row in rows if row.get("error_type"))
        def numeric(name: str) -> list[float]:
            """兼容旧日志或损坏行：聚合时只接受真实数值，不把字符串强制当数字。"""
            return [
                float(row[name]) for row in rows
                if isinstance(row.get(name), (int, float))
            ]

        estimated = numeric("estimated_cost")
        return {
            "window_minutes": minutes,
            "sample_count": len(rows),
            "request_count": len(rows),
            "success_count": sum(row.get("status") == "success" for row in rows),
            "error_count": sum(row.get("status") == "error" for row in rows),
            "error_rate": sum(row.get("status") == "error" for row in rows) / len(rows) if rows else 0.0,
            "average_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "p50_latency_ms": _percentile(latencies, 0.50),
            "p95_latency_ms": _percentile(latencies, 0.95),
            "requests_per_minute": len(rows) / minutes if minutes > 0 else None,
            "input_tokens": sum(numeric("input_tokens")),
            "output_tokens": sum(numeric("output_tokens")),
            "total_tokens": sum(numeric("total_tokens")),
            "estimated_cost": sum(estimated) if estimated else None,
            "cost_note": "estimate only; unknown-priced or unavailable-usage requests are excluded",
            "top_error_types": error_types.most_common(10),
        }


def _print_tree(nodes: list[dict], indent: int = 0) -> None:
    for node in nodes:
        print(
            f"{'  ' * indent}- {node['name']} "
            f"[{node['status']}] {node['duration_ms']:.2f} ms"
        )
        _print_tree(node.get("children", []), indent + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RAG observability report")
    commands = parser.add_subparsers(dest="command", required=True)
    recent = commands.add_parser("recent")
    recent.add_argument("--limit", type=int, default=10)
    trace_parser = commands.add_parser("trace")
    trace_parser.add_argument("--request-id")
    trace_parser.add_argument("--trace-id")
    summary_parser = commands.add_parser("summary")
    summary_parser.add_argument("--minutes", type=int, default=60)
    args = parser.parse_args()
    store = LocalReportStore()
    if args.command == "recent":
        print(json.dumps(store.recent(args.limit), ensure_ascii=False, indent=2))
    elif args.command == "summary":
        print(json.dumps(store.summary(args.minutes), ensure_ascii=False, indent=2))
    else:
        if not args.request_id and not args.trace_id:
            parser.error("trace requires --request-id or --trace-id")
        detail = store.trace_detail(request_id=args.request_id, trace_id=args.trace_id)
        if detail is None:
            raise SystemExit("Trace not found")
        print(json.dumps(detail["request"], ensure_ascii=False, indent=2))
        _print_tree(detail["spans"])


if __name__ == "__main__":
    main()

