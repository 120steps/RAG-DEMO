"""Phase 9 Answer/Citation/Refusal Evaluation，可选调用现有 LLM Judge。"""

from __future__ import annotations

import argparse
import statistics
import time

from ..config import DEFAULT_SETTINGS
from ..enterprise import EnterpriseRAGService
from .answer_eval import (
    RequestPacer,
    build_judge_chain,
    citation_correct,
    deterministic_answer_correct,
)
from .common import RESULTS_DIR, expected_rank, load_test_cases, write_json
from .enterprise_common import EVAL_KNOWLEDGE_BASE, bootstrap_evaluation_knowledge_base


RESULT_FILE = RESULTS_DIR / "phase9_answer_benchmark.json"


def run(*, use_judge: bool = False, limit: int | None = None) -> dict:
    state = bootstrap_evaluation_knowledge_base(DEFAULT_SETTINGS)
    service = EnterpriseRAGService(state["catalog"], DEFAULT_SETTINGS)
    judge_chain = build_judge_chain(DEFAULT_SETTINGS) if use_judge else None
    pacer = RequestPacer(DEFAULT_SETTINGS.gemini_min_interval_seconds)
    details = []
    for case in load_test_cases()[:limit]:
        pacer.wait()
        started = time.perf_counter()
        error = None
        try:
            result = service.chat(
                principal=state["principal"],
                knowledge_base_id=EVAL_KNOWLEDGE_BASE,
                question=case["question"],
            )
        except Exception as exception:
            error = str(exception)
            result = {"answer": "", "refused": False, "sources": [], "documents": [], "retrieved_context": ""}
        latency = (time.perf_counter() - started) * 1000
        judge_result = None
        judge_error = None
        if judge_chain and not error:
            try:
                pacer.wait()
                judged = judge_chain.invoke(
                    {
                        "question": case["question"],
                        "should_answer": case["should_answer"],
                        "expected_answer": case.get("expected_answer"),
                        "expected_keywords": case.get("expected_keywords", []),
                        "context": result.get("retrieved_context", ""),
                        "answer": result.get("answer", ""),
                        "citations": result.get("sources", []),
                    }
                )
                judge_result = judged.model_dump()
            except Exception as exception:
                judge_error = str(exception)
        answer_ok = deterministic_answer_correct(case, result)
        citation_ok = citation_correct(case, result)
        refusal_ok = bool(result.get("refused")) if not case["should_answer"] else None
        rank = expected_rank(case, result.get("documents", [])) if case["should_answer"] else None
        details.append(
            {
                "id": case["id"],
                "question": case["question"],
                "should_answer": case["should_answer"],
                "retrieval_rank": rank,
                "answer": result.get("answer", ""),
                "refused": result.get("refused", False),
                "answer_correct": answer_ok,
                "citation_correct": citation_ok,
                "refusal_correct": refusal_ok,
                "sources": result.get("sources", []),
                "latency_ms": latency,
                "judge_result": judge_result,
                "judge_error": judge_error,
                "error": error,
            }
        )
        print(f"Phase 9 Answer {case['id']}", flush=True)
    answerable = [item for item in details if item["should_answer"]]
    refusal = [item for item in details if not item["should_answer"]]
    judged = [item for item in details if item["judge_result"]]
    summary = {
        "total_cases": len(details),
        "answer_accuracy": sum(item["answer_correct"] is True for item in answerable) / len(answerable) if answerable else 0.0,
        "citation_accuracy": sum(item["citation_correct"] is True for item in answerable) / len(answerable) if answerable else 0.0,
        "refusal_accuracy": sum(item["refusal_correct"] is True for item in refusal) / len(refusal) if refusal else 0.0,
        "faithfulness_average": statistics.fmean(item["judge_result"]["faithfulness_score"] for item in judged) if judged else None,
        "hallucination_rate": sum(item["judge_result"]["hallucination"] for item in judged) / len(judged) if judged else None,
        "average_latency_ms": statistics.fmean(item["latency_ms"] for item in details) if details else 0.0,
        "failed_case_ids": [item["id"] for item in details if item["error"] or item["answer_correct"] is False or item["citation_correct"] is False or item["refusal_correct"] is False],
        "judge_evaluated_cases": len(judged),
    }
    output = {"summary": summary, "details": details}
    write_json(RESULT_FILE, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(run(use_judge=args.judge, limit=args.limit)["summary"])


if __name__ == "__main__":
    main()

