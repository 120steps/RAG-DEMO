"""对已生成的 Phase 9 Answer 做 LLM-as-a-Judge，不重复生成答案。"""

from __future__ import annotations

import json
import statistics

from ..config import DEFAULT_SETTINGS
from ..enterprise import EnterpriseRAGService
from .answer_eval import RequestPacer, build_judge_chain
from .common import RESULTS_DIR, load_test_cases, write_json
from .enterprise_answer_eval import RESULT_FILE as ANSWER_RESULT_FILE
from .enterprise_common import EVAL_KNOWLEDGE_BASE, bootstrap_evaluation_knowledge_base


RESULT_FILE = RESULTS_DIR / "phase9_judge_benchmark.json"


def _context(documents: list[dict]) -> str:
    return "\n\n".join(
        f"[Source: {item.get('source')}, Page: {item.get('page')}, "
        f"Chunk: {item.get('chunk_id')}, Version: {item.get('version_id')}]\n"
        f"{item.get('document', '')}"
        for item in documents
    )


def run() -> dict:
    if not ANSWER_RESULT_FILE.exists():
        raise FileNotFoundError("Run enterprise_answer_eval first")
    answer_output = json.loads(ANSWER_RESULT_FILE.read_text(encoding="utf-8"))
    answers = {item["id"]: item for item in answer_output["details"]}
    state = bootstrap_evaluation_knowledge_base(DEFAULT_SETTINGS)
    service = EnterpriseRAGService(state["catalog"], DEFAULT_SETTINGS)
    judge_chain = build_judge_chain(DEFAULT_SETTINGS)
    pacer = RequestPacer(DEFAULT_SETTINGS.gemini_min_interval_seconds)
    details = []
    for case in load_test_cases():
        stored = answers[case["id"]]
        retrieval = service.retrieve_only(
            principal=state["principal"],
            knowledge_base_id=EVAL_KNOWLEDGE_BASE,
            question=case["question"],
            top_k=3,
        )
        error = None
        result = None
        try:
            pacer.wait()
            judged = judge_chain.invoke(
                {
                    "question": case["question"],
                    "should_answer": case["should_answer"],
                    "expected_answer": case.get("expected_answer"),
                    "expected_keywords": case.get("expected_keywords", []),
                    "context": _context(retrieval.get("documents", [])),
                    "answer": stored.get("answer", ""),
                    "citations": stored.get("sources", []),
                }
            )
            result = judged.model_dump()
        except Exception as exception:
            error = str(exception)
        details.append(
            {
                "id": case["id"],
                "judge_result": result,
                "judge_error": error,
            }
        )
        print(f"Phase 9 Judge {case['id']}", flush=True)
    successful = [item["judge_result"] for item in details if item["judge_result"]]
    summary = {
        "total_cases": len(details),
        "judged_cases": len(successful),
        "judge_failed_cases": [item["id"] for item in details if item["judge_error"]],
        "correctness_average": statistics.fmean(item["correctness_score"] for item in successful) if successful else None,
        "faithfulness_average": statistics.fmean(item["faithfulness_score"] for item in successful) if successful else None,
        "completeness_average": statistics.fmean(item["completeness_score"] for item in successful) if successful else None,
        "hallucination_rate": sum(item["hallucination"] for item in successful) / len(successful) if successful else None,
        "hallucination_case_ids": [detail["id"] for detail in details if detail["judge_result"] and detail["judge_result"]["hallucination"]],
    }
    output = {"summary": summary, "details": details}
    write_json(RESULT_FILE, output)
    return output


if __name__ == "__main__":
    print(run()["summary"])

