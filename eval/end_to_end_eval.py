import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LLM_MODEL
from answer_eval import evaluate_case
from answer_judge import judge_answer
from rag_service import ask_rag


TEST_CASE_FILE = PROJECT_ROOT / "eval" / "test_case.json"
RESULT_FILE = (
    PROJECT_ROOT
    / "eval"
    / "results"
    / "end_to_end_eval.json"
)
GEMINI_REQUEST_INTERVAL_SECONDS = 4.0
MAX_REQUEST_ATTEMPTS = 3


class GeminiRequestLimiter:
    def __init__(self, interval_seconds: float):
        self.interval_seconds = interval_seconds
        self.last_request_completed_at = None

    def wait(self):
        if self.last_request_completed_at is None:
            return

        elapsed = time.perf_counter() - self.last_request_completed_at
        remaining = self.interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def mark_completed(self):
        self.last_request_completed_at = time.perf_counter()


def load_test_cases():
    with TEST_CASE_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_incomplete_details(test_cases):
    if not RESULT_FILE.exists():
        return []

    try:
        with RESULT_FILE.open("r", encoding="utf-8") as file:
            existing_result = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []

    if existing_result.get("parameters", {}).get("completed") is not False:
        return []

    details = existing_result.get("details")
    if not isinstance(details, list):
        return []

    expected_prefix = [
        case["id"] for case in test_cases[:len(details)]
    ]
    actual_prefix = [detail.get("id") for detail in details]
    return details if actual_prefix == expected_prefix else []


def call_rag_with_retry(question, limiter):
    last_error = None

    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        limiter.wait()
        request_started_at = time.perf_counter()
        try:
            result = ask_rag(question)
            measured_total_latency_ms = (
                time.perf_counter() - request_started_at
            ) * 1000
            limiter.mark_completed()
            return result, measured_total_latency_ms, None
        except Exception as error:
            limiter.mark_completed()
            last_error = str(error)
            print(
                f"  RAG attempt {attempt}/{MAX_REQUEST_ATTEMPTS} "
                f"failed: {last_error}"
            )

    return None, None, last_error


def call_judge_with_retry(judge_input, limiter):
    last_result = None

    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        limiter.wait()
        last_result = judge_answer(**judge_input)
        limiter.mark_completed()

        if not last_result["judge_failed"]:
            return last_result

        print(
            f"  Judge attempt {attempt}/{MAX_REQUEST_ATTEMPTS} "
            f"failed: {last_result['judge_error']}"
        )

    return last_result


def build_retrieved_context(result):
    if result is None:
        return []

    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    distances = result.get("distances", [])
    bm25_scores = result.get("bm25_scores", [])
    rerank_scores = result.get("rerank_scores", [])
    fusion_scores = result.get("fusion_scores", [])
    matched_queries = result.get("matched_queries", [])
    context = []

    for index, document in enumerate(documents):
        context.append(
            {
                "document": document,
                "metadata": (
                    metadatas[index]
                    if index < len(metadatas)
                    else {}
                ),
                "distance": (
                    distances[index]
                    if index < len(distances)
                    else None
                ),
                "bm25_score": (
                    bm25_scores[index]
                    if index < len(bm25_scores)
                    else None
                ),
                "rerank_score": (
                    rerank_scores[index]
                    if index < len(rerank_scores)
                    else None
                ),
                "fusion_score": (
                    fusion_scores[index]
                    if index < len(fusion_scores)
                    else None
                ),
                "matched_queries": (
                    matched_queries[index]
                    if index < len(matched_queries)
                    else []
                ),
            }
        )

    return context


def evaluate_end_to_end_case(
    case,
    rag_result,
    measured_total_latency_ms,
    rag_error,
    limiter,
):
    pipeline_total_latency_ms = (
        rag_result.get("total_latency_ms", measured_total_latency_ms)
        if rag_result
        else measured_total_latency_ms
    )
    deterministic = evaluate_case(
        case,
        rag_result,
        pipeline_total_latency_ms or 0.0,
        rag_error,
    )
    retrieved_sources = deterministic["actual_sources"]
    retrieved_context = build_retrieved_context(rag_result)
    retrieval_hit = (
        deterministic["citation_correct"]
        if case["should_answer"]
        else None
    )

    if rag_result is None:
        judge_output = {
            "judge_result": None,
            "raw_judge_result": None,
            "judge_failed": True,
            "judge_error": "RAG request failed; no answer to judge",
        }
    else:
        judge_output = call_judge_with_retry(
            {
                "question": case["question"],
                "answer": rag_result.get("answer", ""),
                "retrieved_context": retrieved_context,
                "expected_answer": case.get("expected_answer"),
                "expected_keywords": case.get(
                    "expected_keywords",
                    [],
                ),
                "should_answer": case["should_answer"],
                "actual_citations": retrieved_sources,
            },
            limiter,
        )

    judge_result = judge_output.get("judge_result") or {}

    return {
        "id": case["id"],
        "question": case["question"],
        "should_answer": case["should_answer"],
        "expected_answer": case.get("expected_answer"),
        "expected_keywords": case.get("expected_keywords", []),
        "expected_source": case.get("expected_source"),
        "expected_page": case.get("expected_page"),
        "retrieval_hit": retrieval_hit,
        "answer": deterministic["answer"],
        "refused": deterministic["refused"],
        "answer_correct_deterministic": deterministic[
            "answer_correct"
        ],
        "citation_correct": deterministic["citation_correct"],
        "refusal_correct": deterministic["refusal_correct"],
        "judge_correctness_score": judge_result.get(
            "correctness_score"
        ),
        "judge_faithfulness_score": judge_result.get(
            "faithfulness_score"
        ),
        "judge_completeness_score": judge_result.get(
            "completeness_score"
        ),
        "hallucination": judge_result.get("hallucination"),
        "judge_reason": judge_result.get("reason"),
        "raw_judge_result": judge_output.get("raw_judge_result"),
        "judge_failed": judge_output.get("judge_failed", True),
        "judge_error": judge_output.get("judge_error"),
        "retrieved_sources": retrieved_sources,
        "retrieved_context": retrieved_context,
        "retrieval_latency_ms": (
            rag_result.get("retrieval_latency_ms")
            if rag_result
            else None
        ),
        "generation_latency_ms": (
            rag_result.get("generation_latency_ms")
            if rag_result
            else None
        ),
        "total_latency_ms": pipeline_total_latency_ms,
        "answer_match_method": deterministic[
            "answer_match_method"
        ],
        "expected_content_matches": deterministic[
            "expected_content_matches"
        ],
        "deterministic_failure_reasons": deterministic[
            "failure_reasons"
        ],
        "rag_error": rag_error,
    }


def safe_average(values):
    usable_values = [value for value in values if value is not None]
    return statistics.mean(usable_values) if usable_values else None


def build_summary(details):
    answerable = [
        detail for detail in details if detail["should_answer"]
    ]
    refusal_cases = [
        detail for detail in details if not detail["should_answer"]
    ]
    judged = [
        detail for detail in details if not detail["judge_failed"]
    ]

    retrieval_hit_count = sum(
        detail["retrieval_hit"] is True for detail in answerable
    )
    deterministic_correct_count = sum(
        detail["answer_correct_deterministic"] is True
        for detail in answerable
    )
    citation_correct_count = sum(
        detail["citation_correct"] is True
        for detail in answerable
    )
    refusal_correct_count = sum(
        detail["refusal_correct"] is True
        for detail in refusal_cases
    )
    hallucination_count = sum(
        detail["hallucination"] is True for detail in judged
    )

    return {
        "total_cases": len(details),
        "answerable_cases": len(answerable),
        "refusal_cases": len(refusal_cases),
        "judged_cases": len(judged),
        "judge_failed_case_ids": [
            detail["id"]
            for detail in details
            if detail["judge_failed"]
        ],
        "retrieval_hit_count": retrieval_hit_count,
        "retrieval_hit_rate": (
            retrieval_hit_count / len(answerable)
            if answerable
            else 0.0
        ),
        "deterministic_answer_correct_count": (
            deterministic_correct_count
        ),
        "deterministic_answer_accuracy": (
            deterministic_correct_count / len(answerable)
            if answerable
            else 0.0
        ),
        "citation_correct_count": citation_correct_count,
        "citation_accuracy": (
            citation_correct_count / len(answerable)
            if answerable
            else 0.0
        ),
        "refusal_correct_count": refusal_correct_count,
        "refusal_accuracy": (
            refusal_correct_count / len(refusal_cases)
            if refusal_cases
            else 0.0
        ),
        "judge_correctness_average": safe_average(
            [detail["judge_correctness_score"] for detail in judged]
        ),
        "judge_faithfulness_average": safe_average(
            [detail["judge_faithfulness_score"] for detail in judged]
        ),
        "judge_completeness_average": safe_average(
            [detail["judge_completeness_score"] for detail in judged]
        ),
        "hallucination_count": hallucination_count,
        "hallucination_rate": (
            hallucination_count / len(judged) if judged else 0.0
        ),
        "average_end_to_end_latency_ms": safe_average(
            [detail["total_latency_ms"] for detail in details]
        ),
    }


def save_result(details, completed):
    output = {
        "parameters": {
            "test_case_file": str(
                TEST_CASE_FILE.relative_to(PROJECT_ROOT)
            ),
            "rag_pipeline": "rag_service.ask_rag",
            "judge_model": LLM_MODEL,
            "gemini_request_interval_seconds": (
                GEMINI_REQUEST_INTERVAL_SECONDS
            ),
            "max_request_attempts": MAX_REQUEST_ATTEMPTS,
            "retrieval_hit_denominator": "should_answer=true cases",
            "latency_excludes_judge": True,
            "completed": completed,
        },
        "summary": build_summary(details),
        "details": details,
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return output


def print_summary(summary):
    def format_average(value, suffix=""):
        return f"{value:.3f}{suffix}" if value is not None else "N/A"

    print()
    print("End-to-End RAG Benchmark")
    print("=" * 68)
    print(f"Total Cases: {summary['total_cases']}")
    print(
        "Retrieval Hit Rate: "
        f"{summary['retrieval_hit_count']}/"
        f"{summary['answerable_cases']} "
        f"({summary['retrieval_hit_rate']:.2%})"
    )
    print(
        "Deterministic Answer Accuracy: "
        f"{summary['deterministic_answer_correct_count']}/"
        f"{summary['answerable_cases']} "
        f"({summary['deterministic_answer_accuracy']:.2%})"
    )
    print(
        "Citation Accuracy: "
        f"{summary['citation_correct_count']}/"
        f"{summary['answerable_cases']} "
        f"({summary['citation_accuracy']:.2%})"
    )
    print(
        "Refusal Accuracy: "
        f"{summary['refusal_correct_count']}/"
        f"{summary['refusal_cases']} "
        f"({summary['refusal_accuracy']:.2%})"
    )
    print(
        "Judge Correctness Average: "
        f"{format_average(summary['judge_correctness_average'])}"
    )
    print(
        "Judge Faithfulness Average: "
        f"{format_average(summary['judge_faithfulness_average'])}"
    )
    print(
        "Judge Completeness Average: "
        f"{format_average(summary['judge_completeness_average'])}"
    )
    print(
        "Hallucination Rate: "
        f"{summary['hallucination_count']}/"
        f"{summary['judged_cases']} "
        f"({summary['hallucination_rate']:.2%})"
    )
    print(
        "Average End-to-End Latency: "
        f"{format_average(summary['average_end_to_end_latency_ms'], ' ms')}"
    )
    print(
        f"Judge Failed Case IDs: "
        f"{summary['judge_failed_case_ids']}"
    )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    test_cases = load_test_cases()
    limiter = GeminiRequestLimiter(GEMINI_REQUEST_INTERVAL_SECONDS)
    details = load_incomplete_details(test_cases)

    if details:
        print(
            f"Resuming incomplete result after "
            f"{details[-1]['id']} ({len(details)}/{len(test_cases)}).",
            flush=True,
        )

    for index in range(len(details), len(test_cases)):
        case = test_cases[index]
        print(
            f"Running {case['id']} "
            f"({index + 1}/{len(test_cases)})...",
            flush=True,
        )
        rag_result, measured_latency_ms, rag_error = (
            call_rag_with_retry(case["question"], limiter)
        )
        detail = evaluate_end_to_end_case(
            case,
            rag_result,
            measured_latency_ms,
            rag_error,
            limiter,
        )
        details.append(detail)
        save_result(details, completed=False)
        print(
            f"  retrieval_hit={detail['retrieval_hit']}, "
            "deterministic_correct="
            f"{detail['answer_correct_deterministic']}, "
            "judge_correctness="
            f"{detail['judge_correctness_score']}, "
            f"hallucination={detail['hallucination']}",
            flush=True,
        )

    output = save_result(details, completed=True)
    print_summary(output["summary"])
    print(
        "Result saved: "
        f"{RESULT_FILE.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
