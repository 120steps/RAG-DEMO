"""V3 answer evaluation with deterministic checks and optional LLM judge."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import unicodedata
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..chain import NativeRAGService, get_chat_model
from ..config import DEFAULT_SETTINGS
from .common import RESULTS_DIR, expected_rank, load_test_cases, write_json


RESULT_FILE = RESULTS_DIR / "answer_benchmark.json"
PIPELINE_CONTRACT = "structured_answer_v1"
REFUSAL_INDICATORS = (
    "无法回答", "不能回答", "未说明", "没有说明", "未提供", "无法确定",
)


class JudgeResult(BaseModel):
    correctness_score: int = Field(ge=0, le=2)
    faithfulness_score: int = Field(ge=0, le=2)
    completeness_score: int = Field(ge=0, le=2)
    hallucination: bool
    reason: str


JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是 RAG 答案评估器，不是问答助手。只能根据问题、期望内容和 Retrieved Context "
            "评价 Generated Answer，禁止补充外部知识。Correctness/faithfulness/completeness 均为 "
            "0=错误或缺失、1=部分、2=完整正确。若答案包含 Context 不支持的实质主张，"
            "hallucination=true。should_answer=false 时，正确拒答应视为正确且忠实。",
        ),
        (
            "human",
            "Question: {question}\nShould answer: {should_answer}\n"
            "Expected answer: {expected_answer}\nExpected keywords: {expected_keywords}\n"
            "Retrieved Context:\n{context}\n\nGenerated Answer:\n{answer}\n"
            "Actual citations: {citations}",
        ),
    ]
)


class RequestPacer:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.last = 0.0

    def wait(self) -> None:
        remaining = self.seconds - (time.monotonic() - self.last)
        if remaining > 0:
            time.sleep(remaining)
        self.last = time.monotonic()


def normalize_text(text: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    normalized = normalized.replace("不得", "不")
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalized)


def _bigram_recall(expected: str, actual: str) -> float:
    if len(expected) < 2:
        return 1.0 if expected in actual else 0.0
    bigrams = {expected[index:index + 2] for index in range(len(expected) - 1)}
    return sum(item in actual for item in bigrams) / len(bigrams)


def match_content(expected: str, answer: str) -> bool:
    expected_normalized = normalize_text(expected)
    answer_normalized = normalize_text(answer)
    if not expected_normalized:
        return False
    if expected_normalized in answer_normalized:
        return True
    numbers = re.findall(r"\d+(?:\.\d+)?", expected_normalized)
    if any(number not in answer_normalized for number in numbers):
        return False
    latin = re.findall(r"[a-z][a-z0-9.]*", expected_normalized)
    if any(entity not in answer_normalized for entity in latin):
        return False
    return _bigram_recall(expected_normalized, answer_normalized) >= (
        0.50 if numbers else 0.70
    )


def deterministic_answer_correct(case: dict, result: dict) -> bool | None:
    if not case["should_answer"]:
        return None
    if result.get("refused"):
        return False
    answer = result.get("answer", "")
    normalized = normalize_text(answer)
    if any(normalize_text(item) in normalized for item in REFUSAL_INDICATORS):
        return False
    keywords = case.get("expected_keywords") or []
    expected_items = keywords or [case.get("expected_answer")]
    expected_items = [item for item in expected_items if item]
    return bool(expected_items) and all(
        match_content(item, answer) for item in expected_items
    )


def citation_correct(case: dict, result: dict) -> bool | None:
    if not case["should_answer"]:
        return None
    return any(
        citation.get("source") == case.get("expected_source")
        and citation.get("page") == case.get("expected_page")
        for citation in result.get("sources", [])
    )


def build_judge_chain(settings=DEFAULT_SETTINGS):
    return JUDGE_PROMPT | get_chat_model(settings).with_structured_output(JudgeResult)


def _summary(details: list[dict]) -> dict:
    answerable = [item for item in details if item["should_answer"]]
    refusal = [item for item in details if not item["should_answer"]]
    judged = [item for item in details if item.get("judge_result")]
    latencies = [item["total_latency_ms"] for item in details]
    return {
        "total_cases": len(details),
        "answered_cases": len(answerable),
        "refusal_cases": len(refusal),
        "retrieval_hit_rate": sum(bool(item["retrieval_hit"]) for item in answerable) / len(answerable) if answerable else 0.0,
        "answer_correctness": sum(item["answer_correct"] is True for item in answerable) / len(answerable) if answerable else 0.0,
        "citation_correctness": sum(item["citation_correct"] is True for item in answerable) / len(answerable) if answerable else 0.0,
        "refusal_correctness": sum(item["refusal_correct"] is True for item in refusal) / len(refusal) if refusal else 0.0,
        "judge_correctness_average": statistics.fmean(item["judge_result"]["correctness_score"] for item in judged) if judged else None,
        "judge_faithfulness_average": statistics.fmean(item["judge_result"]["faithfulness_score"] for item in judged) if judged else None,
        "judge_completeness_average": statistics.fmean(item["judge_result"]["completeness_score"] for item in judged) if judged else None,
        "hallucination_rate": sum(item["judge_result"]["hallucination"] for item in judged) / len(judged) if judged else None,
        "average_end_to_end_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "failed_case_ids": [item["id"] for item in details if item["failure_reasons"]],
        "judge_failed_case_ids": [item["id"] for item in details if item.get("judge_error")],
    }


def run_answer_evaluation(
    *,
    use_judge: bool = True,
    limit: int | None = None,
    rerun_failed: bool = False,
    force: bool = False,
) -> dict:
    service = NativeRAGService(settings=DEFAULT_SETTINGS)
    judge_chain = build_judge_chain() if use_judge else None
    pacer = RequestPacer(DEFAULT_SETTINGS.gemini_min_interval_seconds)
    cases = load_test_cases()[:limit]
    details = []
    indices_to_run = list(range(len(cases)))
    if RESULT_FILE.exists() and not force:
        try:
            previous = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
            previous_details = previous.get("details", [])
            expected_ids = [case["id"] for case in cases[:len(previous_details)]]
            if (
                previous.get("completed") is False
                and [item.get("id") for item in previous_details] == expected_ids
            ):
                details = previous_details
                print(f"Resuming at {len(details)}/{len(cases)} cases", flush=True)
                indices_to_run = list(range(len(details), len(cases)))
            elif (
                rerun_failed
                and previous.get("completed") is True
                and len(previous_details) == len(cases)
            ):
                details = previous_details
                indices_to_run = [
                    index
                    for index, item in enumerate(details)
                    if item.get("failure_reasons") or item.get("judge_error")
                ]
                print(
                    f"Rerunning selected cases: "
                    f"{[cases[index]['id'] for index in indices_to_run]}",
                    flush=True,
                )
        except (OSError, json.JSONDecodeError):
            details = []
    for index in indices_to_run:
        case = cases[index]
        print(f"Answer {case['id']} ({index + 1}/{len(cases)})", flush=True)
        pacer.wait()
        started = time.perf_counter()
        error = None
        try:
            result = service.ask_rag(case["question"])
        except Exception as exc:
            result = {
                "answer": "", "refused": False, "sources": [],
                "documents": [], "retrieved_context": "",
                "retrieval_latency_ms": 0.0,
            }
            error = str(exc)
        generation_done = time.perf_counter()

        judge_result = None
        judge_error = None
        judge_latency_ms = None
        if judge_chain is not None and error is None:
            try:
                pacer.wait()
                judge_started = time.perf_counter()
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
                judge_latency_ms = (time.perf_counter() - judge_started) * 1000
            except Exception as exc:
                judge_error = str(exc)

        documents = result.get("documents", [])
        rank = expected_rank(case, documents) if case["should_answer"] else None
        answer_ok = deterministic_answer_correct(case, result)
        citation_ok = citation_correct(case, result)
        refusal_ok = (
            bool(result.get("refused")) if not case["should_answer"] else None
        )
        reasons = []
        if error:
            reasons.append("request_failed")
        if answer_ok is False:
            reasons.append("answer_incorrect")
        if citation_ok is False:
            reasons.append("citation_incorrect")
        if refusal_ok is False:
            reasons.append("refusal_incorrect")
        detail = {
                "id": case["id"],
                "question": case["question"],
                "should_answer": case["should_answer"],
                "retrieval_hit": rank is not None if case["should_answer"] else None,
                "retrieval_rank": rank,
                "answer": result.get("answer", ""),
                "refused": result.get("refused", False),
                "answer_correct": answer_ok,
                "citation_correct": citation_ok,
                "refusal_correct": refusal_ok,
                "expected_answer": case.get("expected_answer"),
                "expected_keywords": case.get("expected_keywords", []),
                "expected_source": case.get("expected_source"),
                "expected_page": case.get("expected_page"),
                "actual_sources": result.get("sources", []),
                "retrieved_context": result.get("retrieved_context", ""),
                "retrieved_documents": documents,
                "retrieval_latency_ms": result.get("retrieval_latency_ms", 0.0),
                "generation_latency_ms": (generation_done - started) * 1000 - result.get("retrieval_latency_ms", 0.0),
                "total_latency_ms": (generation_done - started) * 1000,
                "judge_latency_ms": judge_latency_ms,
                "judge_result": judge_result,
                "judge_error": judge_error,
                "evaluation_error": error,
                "failure_reasons": reasons,
            }
        if index < len(details):
            details[index] = detail
        else:
            details.append(detail)
        write_json(
            RESULT_FILE,
            {
                "completed": False,
                "valid_for_final_pipeline": False,
                "pipeline_contract": PIPELINE_CONTRACT,
                "summary": _summary(details),
                "details": details,
            },
        )
    output = {
        "completed": True,
        "valid_for_final_pipeline": True,
        "pipeline_contract": PIPELINE_CONTRACT,
        "summary": _summary(details),
        "details": details,
    }
    write_json(RESULT_FILE, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = run_answer_evaluation(
        use_judge=not args.no_judge,
        limit=args.limit,
        rerun_failed=args.rerun_failed,
        force=args.force,
    )
    print(output["summary"])


if __name__ == "__main__":
    main()

