"""Phase 12 质量门禁：比较真实评估 JSON、受控基线与明确阈值。

脚本只读取结果，不修改 Ground Truth。指标缺失、样本不足或低于阈值都会返回
非零退出码；因此 CI 不会把“没有执行评估”误写成通过。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def _rate(summary: dict[str, Any], name: str) -> float:
    value = summary.get(name)
    if isinstance(value, dict):
        value = value.get("hit_rate")
    if value is None:
        raise ValueError(f"missing metric: {name}")
    return float(value)


def evaluate_gate(
    retrieval: dict, answer: dict, security: dict, thresholds: dict, baseline: dict
) -> dict:
    """返回逐项 Gate 结果；任何关键指标缺失都会抛出 ValueError。"""
    rs = retrieval.get("summary", retrieval)
    ans = answer.get("summary", answer)
    sec = security.get("summary", security)
    checks: list[dict[str, Any]] = []

    def minimum(name: str, current: float, threshold: float) -> None:
        checks.append(
            {
                "metric": name,
                "current": current,
                "threshold": threshold,
                "result": "PASS" if current >= threshold else "FAIL",
            }
        )

    total = int(rs.get("total_cases", 0))
    minimum("total_retrieval_cases", total, thresholds["minimum_total_retrieval_cases"])
    for k in (1, 3, 5, 10):
        minimum(f"hit_at_{k}", _rate(rs, f"hit_at_{k}"), thresholds[f"minimum_hit_at_{k}"])
    latency = float(rs["average_retrieval_latency_ms"])
    base_latency = float(baseline["retrieval"]["average_latency_ms"])
    max_latency = base_latency * (1 + thresholds["maximum_latency_regression_ratio"])
    checks.append(
        {
            "metric": "average_retrieval_latency_ms",
            "current": latency,
            "threshold": max_latency,
            "result": "PASS" if latency <= max_latency else "FAIL",
        }
    )
    minimum("answer_accuracy", float(ans["answer_accuracy"]), thresholds["minimum_answer_accuracy"])
    minimum(
        "citation_accuracy",
        float(ans["citation_accuracy"]),
        thresholds["minimum_citation_correctness"],
    )
    minimum(
        "refusal_accuracy",
        float(ans["refusal_accuracy"]),
        thresholds["minimum_refusal_correctness"],
    )
    failed = int(sec["failed"])
    checks.append(
        {
            "metric": "security_failures",
            "current": failed,
            "threshold": thresholds["required_security_failures"],
            "result": "PASS" if failed == thresholds["required_security_failures"] else "FAIL",
        }
    )
    return {
        "result": "PASS" if all(x["result"] == "PASS" for x in checks) else "FAIL",
        "checks": checks,
        "baseline_provenance": baseline["provenance"],
    }


def markdown_report(report: dict) -> str:
    rows = [
        "# RAG Quality Gate",
        "",
        f"Final: **{report['result']}**",
        "",
        "| Metric | Current | Threshold | Result |",
        "|---|---:|---:|---|",
    ]
    rows.extend(
        f"| {c['metric']} | {c['current']} | {c['threshold']} | {c['result']} |"
        for c in report["checks"]
    )
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--security", required=True)
    parser.add_argument("--output", default=str(HERE / "results" / "quality_gate.json"))
    args = parser.parse_args()

    def load(path: str | Path) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    try:
        report = evaluate_gate(
            load(args.retrieval),
            load(args.answer),
            load(args.security),
            load(HERE / "quality_gate.json"),
            load(HERE / "baselines" / "v3_phase9.json"),
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"QUALITY GATE BLOCKED: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
