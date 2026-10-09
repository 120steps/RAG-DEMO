from __future__ import annotations

import pytest

from rag_langchain_native.eval.quality_gate import evaluate_gate


def _inputs():
    thresholds = {
        "minimum_total_retrieval_cases": 2,
        "minimum_hit_at_1": 0.5,
        "minimum_hit_at_3": 1,
        "minimum_hit_at_5": 1,
        "minimum_hit_at_10": 1,
        "maximum_latency_regression_ratio": 0.25,
        "minimum_answer_accuracy": 0.9,
        "minimum_citation_correctness": 1,
        "minimum_refusal_correctness": 1,
        "required_security_failures": 0,
    }
    baseline = {"provenance": {}, "retrieval": {"average_latency_ms": 100}}
    retrieval = {
        "total_cases": 2,
        "average_retrieval_latency_ms": 110,
        **{f"hit_at_{k}": {"hit_rate": 1} for k in (1, 3, 5, 10)},
    }
    answer = {"answer_accuracy": 1, "citation_accuracy": 1, "refusal_accuracy": 1}
    return retrieval, answer, {"failed": 0}, thresholds, baseline


def test_quality_gate_passes_complete_metrics():
    assert evaluate_gate(*_inputs())["result"] == "PASS"


def test_quality_gate_fails_regression_and_blocks_missing_metric():
    values = list(_inputs())
    values[0]["hit_at_1"] = {"hit_rate": 0}
    assert evaluate_gate(*values)["result"] == "FAIL"
    values = list(_inputs())
    del values[1]["citation_accuracy"]
    with pytest.raises(KeyError):
        evaluate_gate(*values)
