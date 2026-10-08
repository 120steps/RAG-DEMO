import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    LLM_MODEL,
    QUERY_EXPANSION_ENABLED,
    QUERY_REWRITE_ENABLED,
    TOP_K,
)
from end_to_end_eval import (
    GEMINI_REQUEST_INTERVAL_SECONDS,
    MAX_REQUEST_ATTEMPTS,
    GeminiRequestLimiter,
    build_summary as build_answer_summary,
    evaluate_end_to_end_case,
)
from langchain_rag.lc_full_rag import (
    ask_langchain_full_rag,
    retrieve_advanced,
    serialize_candidates,
)
from langchain_rag.lc_vectorstore import (
    SOURCE_CHROMA_DIR,
    SOURCE_COLLECTION_NAME,
)
from query_expander import expand_query
from query_rewriter import rewrite_query
from rag_service import ask_rag
from reranker import RERANKER_MODEL
from result_fusion import RRF_K
from retrieval import (
    BM25_RETRIEVAL_K,
    RERANK_CANDIDATE_K,
    VECTOR_RETRIEVAL_K,
    retrieve_candidates,
)


TEST_CASE_FILE = PROJECT_ROOT / "eval" / "test_case.json"
RESULT_FILE = (
    PROJECT_ROOT
    / "eval"
    / "results"
    / "langchain_full_benchmark.json"
)
RETRIEVAL_TOP_K = 10
HIT_K_VALUES = (1, 3, 5, 10)
USE_HYBRID = True
USE_RERANKER = True


def load_test_cases() -> list[dict[str, Any]]:
    with TEST_CASE_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_retrieval_test_cases() -> list[dict[str, Any]]:
    return [
        case
        for case in load_test_cases()
        if case.get("should_answer", True)
        and case.get("expected_source") is not None
        and case.get("expected_page") is not None
    ]


def retrieve_manual(
    question: str,
    collection,
    *,
    final_top_k: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    retrieval_query = (
        rewrite_query(question)
        if QUERY_REWRITE_ENABLED
        else question
    )
    expanded_queries = (
        expand_query(retrieval_query)
        if QUERY_EXPANSION_ENABLED
        else [retrieval_query]
    )
    candidates = retrieve_candidates(
        retrieval_query,
        collection,
        use_hybrid=USE_HYBRID,
        use_reranker=USE_RERANKER,
        bm25_index=None,
        final_top_k=final_top_k,
        expanded_queries=(
            expanded_queries
            if QUERY_EXPANSION_ENABLED
            else None
        ),
    )
    return {
        "original_query": question,
        "retrieval_query": retrieval_query,
        "expanded_queries": expanded_queries,
        "candidates": candidates,
        "latency_ms": (time.perf_counter() - started_at) * 1000,
    }


def find_hit_rank(
    candidates: list[dict[str, Any]],
    expected_source: str,
    expected_page: int,
) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        metadata = candidate["metadata"]
        if (
            metadata.get("source") == expected_source
            and metadata.get("page") == expected_page
        ):
            return rank
    return None


def candidate_identity(candidate: dict[str, Any]) -> tuple[Any, Any, Any]:
    metadata = candidate["metadata"]
    return (
        metadata.get("source"),
        metadata.get("page"),
        metadata.get("chunk_id"),
    )


def build_hit_metrics(
    hit_ranks: list[int | None],
    latencies_ms: list[float],
) -> dict[str, Any]:
    total_cases = len(hit_ranks)
    summary: dict[str, Any] = {
        "total_cases": total_cases,
        "average_latency_ms": (
            statistics.fmean(latencies_ms) if latencies_ms else 0.0
        ),
    }
    for top_k in HIT_K_VALUES:
        hit_count = sum(
            rank is not None and rank <= top_k
            for rank in hit_ranks
        )
        summary[f"hit_at_{top_k}"] = {
            "hit_count": hit_count,
            "hit_rate": hit_count / total_cases if total_cases else 0.0,
        }
    return summary


def run_retrieval_comparison() -> dict[str, Any]:
    collection = chromadb.PersistentClient(
        path=str(SOURCE_CHROMA_DIR)
    ).get_collection(name=SOURCE_COLLECTION_NAME)
    test_cases = load_retrieval_test_cases()
    details = []
    manual_ranks = []
    langchain_ranks = []
    manual_latencies = []
    langchain_latencies = []

    for case in test_cases:
        manual = retrieve_manual(
            case["question"],
            collection,
            final_top_k=RETRIEVAL_TOP_K,
        )
        langchain_started_at = time.perf_counter()
        langchain = retrieve_advanced(
            case["question"],
            final_top_k=RETRIEVAL_TOP_K,
            use_hybrid=USE_HYBRID,
            use_reranker=USE_RERANKER,
        )
        langchain_measured_latency = (
            time.perf_counter() - langchain_started_at
        ) * 1000

        manual_rank = find_hit_rank(
            manual["candidates"],
            case["expected_source"],
            case["expected_page"],
        )
        langchain_rank = find_hit_rank(
            langchain["candidates"],
            case["expected_source"],
            case["expected_page"],
        )
        manual_ids = [
            candidate_identity(candidate)
            for candidate in manual["candidates"]
        ]
        langchain_ids = [
            candidate_identity(candidate)
            for candidate in langchain["candidates"]
        ]

        manual_ranks.append(manual_rank)
        langchain_ranks.append(langchain_rank)
        manual_latencies.append(manual["latency_ms"])
        langchain_latencies.append(langchain_measured_latency)
        details.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_source": case["expected_source"],
                "expected_page": case["expected_page"],
                "manual_hit_rank": manual_rank,
                "langchain_hit_rank": langchain_rank,
                "rank_classification": classify_rank(
                    manual_rank,
                    langchain_rank,
                ),
                "candidate_order_match": manual_ids == langchain_ids,
                "manual_retrieval_query": manual["retrieval_query"],
                "langchain_retrieval_query": langchain[
                    "retrieval_query"
                ],
                "manual_expanded_queries": manual["expanded_queries"],
                "langchain_expanded_queries": langchain[
                    "expanded_queries"
                ],
                "manual_latency_ms": manual["latency_ms"],
                "langchain_latency_ms": langchain_measured_latency,
                "manual_top10": serialize_candidates(
                    manual["candidates"]
                ),
                "langchain_top10": serialize_candidates(
                    langchain["candidates"]
                ),
            }
        )
        print(
            f"Retrieval {case['id']}: manual={manual_rank}, "
            f"langchain={langchain_rank}, "
            f"order_match={manual_ids == langchain_ids}",
            flush=True,
        )

    degraded_cases = [
        detail["id"]
        for detail in details
        if detail["rank_classification"] == "degraded"
    ]
    return {
        "manual_summary": build_hit_metrics(
            manual_ranks,
            manual_latencies,
        ),
        "langchain_summary": build_hit_metrics(
            langchain_ranks,
            langchain_latencies,
        ),
        "degraded_cases": degraded_cases,
        "candidate_order_difference_cases": [
            detail["id"]
            for detail in details
            if not detail["candidate_order_match"]
        ],
        "details": details,
    }


def classify_rank(
    reference_rank: int | None,
    current_rank: int | None,
) -> str:
    if reference_rank == current_rank:
        return "unchanged"
    if reference_rank is None:
        return "improved"
    if current_rank is None:
        return "degraded"
    return "improved" if current_rank < reference_rank else "degraded"


def call_pipeline_with_retry(
    pipeline: Callable[[str], dict[str, Any]],
    question: str,
    limiter: GeminiRequestLimiter,
) -> tuple[dict[str, Any] | None, float | None, str | None]:
    last_error = None
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        limiter.wait()
        started_at = time.perf_counter()
        try:
            result = pipeline(question)
            measured_latency_ms = (
                time.perf_counter() - started_at
            ) * 1000
            limiter.mark_completed()
            return result, measured_latency_ms, None
        except Exception as error:
            limiter.mark_completed()
            last_error = str(error)
            print(
                f"  RAG attempt {attempt}/{MAX_REQUEST_ATTEMPTS} "
                f"failed: {last_error}",
                flush=True,
            )
    return None, None, last_error


def build_answer_comparisons(
    manual_details: list[dict[str, Any]],
    langchain_details: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons = []
    degraded_cases = []

    for manual, langchain in zip(manual_details, langchain_details):
        reasons = []
        if (
            manual["answer_correct_deterministic"] is True
            and langchain["answer_correct_deterministic"] is not True
        ):
            reasons.append("deterministic_answer_correctness")
        if (
            manual["citation_correct"] is True
            and langchain["citation_correct"] is not True
        ):
            reasons.append("citation_correctness")
        if (
            manual["refusal_correct"] is True
            and langchain["refusal_correct"] is not True
        ):
            reasons.append("refusal_correctness")

        manual_correctness = manual["judge_correctness_score"]
        langchain_correctness = langchain["judge_correctness_score"]
        if (
            manual_correctness is not None
            and langchain_correctness is not None
            and langchain_correctness < manual_correctness
        ):
            reasons.append("judge_correctness")

        manual_faithfulness = manual["judge_faithfulness_score"]
        langchain_faithfulness = langchain[
            "judge_faithfulness_score"
        ]
        if (
            manual_faithfulness is not None
            and langchain_faithfulness is not None
            and langchain_faithfulness < manual_faithfulness
        ):
            reasons.append("judge_faithfulness")

        comparison = {
            "id": manual["id"],
            "degraded": bool(reasons),
            "degradation_reasons": reasons,
            "manual_answer_correct": manual[
                "answer_correct_deterministic"
            ],
            "langchain_answer_correct": langchain[
                "answer_correct_deterministic"
            ],
            "manual_citation_correct": manual["citation_correct"],
            "langchain_citation_correct": langchain[
                "citation_correct"
            ],
            "manual_refusal_correct": manual["refusal_correct"],
            "langchain_refusal_correct": langchain[
                "refusal_correct"
            ],
            "manual_judge_correctness": manual_correctness,
            "langchain_judge_correctness": langchain_correctness,
            "manual_judge_faithfulness": manual_faithfulness,
            "langchain_judge_faithfulness": langchain_faithfulness,
        }
        comparisons.append(comparison)
        if reasons:
            degraded_cases.append(
                {
                    "id": manual["id"],
                    "reasons": reasons,
                }
            )

    return comparisons, degraded_cases


def load_resume_details(
    test_cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not RESULT_FILE.exists():
        return [], []
    try:
        with RESULT_FILE.open("r", encoding="utf-8") as file:
            existing = json.load(file)
    except (OSError, json.JSONDecodeError):
        return [], []

    parameters = existing.get("parameters", {})
    if parameters.get("completed") is not False:
        return [], []
    answer_evaluation = existing.get("answer_evaluation", {})
    manual = answer_evaluation.get("manual_details", [])
    langchain = answer_evaluation.get("langchain_details", [])
    if not isinstance(manual, list) or not isinstance(langchain, list):
        return [], []
    if len(manual) != len(langchain):
        return [], []
    expected_ids = [case["id"] for case in test_cases[:len(manual)]]
    if (
        [detail.get("id") for detail in manual] != expected_ids
        or [detail.get("id") for detail in langchain] != expected_ids
    ):
        return [], []
    return manual, langchain


def save_benchmark(
    retrieval_comparison: dict[str, Any],
    manual_details: list[dict[str, Any]],
    langchain_details: list[dict[str, Any]],
    *,
    completed: bool,
) -> dict[str, Any]:
    answer_comparisons, answer_degraded_cases = (
        build_answer_comparisons(manual_details, langchain_details)
    )
    output = {
        "parameters": {
            "completed": completed,
            "test_case_file": str(
                TEST_CASE_FILE.relative_to(PROJECT_ROOT)
            ),
            "collection_name": SOURCE_COLLECTION_NAME,
            "collection_directory": str(
                SOURCE_CHROMA_DIR.relative_to(PROJECT_ROOT)
            ),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "answer_top_k": TOP_K,
            "retrieval_evaluation_top_k": RETRIEVAL_TOP_K,
            "vector_retrieval_k": VECTOR_RETRIEVAL_K,
            "bm25_retrieval_k": BM25_RETRIEVAL_K,
            "rerank_candidate_k": RERANK_CANDIDATE_K,
            "rrf_k": RRF_K,
            "reranker_model": RERANKER_MODEL,
            "llm_model": LLM_MODEL,
            "use_hybrid": USE_HYBRID,
            "use_reranker": USE_RERANKER,
            "query_rewrite_enabled": QUERY_REWRITE_ENABLED,
            "query_expansion_enabled": QUERY_EXPANSION_ENABLED,
            "gemini_request_interval_seconds": (
                GEMINI_REQUEST_INTERVAL_SECONDS
            ),
            "evaluation_methods": [
                "eval.answer_eval.evaluate_case",
                "eval.answer_judge.judge_answer",
                "eval.end_to_end_eval.evaluate_end_to_end_case",
            ],
        },
        "retrieval_comparison": retrieval_comparison,
        "answer_evaluation": {
            "manual_summary": build_answer_summary(manual_details),
            "langchain_summary": build_answer_summary(
                langchain_details
            ),
            "degraded_cases": answer_degraded_cases,
            "case_comparisons": answer_comparisons,
            "manual_details": manual_details,
            "langchain_details": langchain_details,
        },
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return output


def print_retrieval_summary(retrieval: dict[str, Any]) -> None:
    manual = retrieval["manual_summary"]
    langchain = retrieval["langchain_summary"]
    print("\nFull Retrieval Comparison")
    print("=" * 76)
    print(f"{'Metric':<12}{'Manual':<30}{'LangChain':<30}")
    for top_k in HIT_K_VALUES:
        key = f"hit_at_{top_k}"
        manual_metric = manual[key]
        langchain_metric = langchain[key]
        manual_text = (
            f"{manual_metric['hit_count']}/{manual['total_cases']} "
            f"({manual_metric['hit_rate']:.2%})"
        )
        langchain_text = (
            f"{langchain_metric['hit_count']}/"
            f"{langchain['total_cases']} "
            f"({langchain_metric['hit_rate']:.2%})"
        )
        print(
            f"Hit@{top_k:<7}{manual_text:<30}{langchain_text:<30}"
        )
    print(
        f"{'Avg latency':<12}"
        f"{manual['average_latency_ms']:<30.2f}"
        f"{langchain['average_latency_ms']:<30.2f}"
    )
    print(f"Retrieval degraded: {retrieval['degraded_cases']}")


def print_answer_summary(output: dict[str, Any]) -> None:
    answer_evaluation = output["answer_evaluation"]
    manual = answer_evaluation["manual_summary"]
    langchain = answer_evaluation["langchain_summary"]

    print("\nFull Answer Comparison")
    print("=" * 76)
    rows = (
        (
            "Answer accuracy",
            manual["deterministic_answer_accuracy"],
            langchain["deterministic_answer_accuracy"],
            "percent",
        ),
        (
            "Citation accuracy",
            manual["citation_accuracy"],
            langchain["citation_accuracy"],
            "percent",
        ),
        (
            "Refusal accuracy",
            manual["refusal_accuracy"],
            langchain["refusal_accuracy"],
            "percent",
        ),
        (
            "Judge correctness",
            manual["judge_correctness_average"],
            langchain["judge_correctness_average"],
            "score",
        ),
        (
            "Judge faithfulness",
            manual["judge_faithfulness_average"],
            langchain["judge_faithfulness_average"],
            "score",
        ),
        (
            "Average latency ms",
            manual["average_end_to_end_latency_ms"],
            langchain["average_end_to_end_latency_ms"],
            "number",
        ),
    )
    print(f"{'Metric':<24}{'Manual':<24}{'LangChain':<24}")
    for name, manual_value, langchain_value, value_type in rows:
        if value_type == "percent":
            manual_text = f"{manual_value:.2%}"
            langchain_text = f"{langchain_value:.2%}"
        elif value_type == "score":
            manual_text = (
                f"{manual_value:.3f}/2"
                if manual_value is not None
                else "N/A"
            )
            langchain_text = (
                f"{langchain_value:.3f}/2"
                if langchain_value is not None
                else "N/A"
            )
        else:
            manual_text = f"{manual_value:.2f}"
            langchain_text = f"{langchain_value:.2f}"
        print(f"{name:<24}{manual_text:<24}{langchain_text:<24}")
    print(f"Answer degraded: {answer_evaluation['degraded_cases']}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Running retrieval comparison...", flush=True)
    retrieval_comparison = run_retrieval_comparison()
    print_retrieval_summary(retrieval_comparison)

    test_cases = load_test_cases()
    manual_details, langchain_details = load_resume_details(test_cases)
    if manual_details:
        print(
            f"Resuming answer evaluation at "
            f"{len(manual_details)}/{len(test_cases)} cases.",
            flush=True,
        )
    limiter = GeminiRequestLimiter(GEMINI_REQUEST_INTERVAL_SECONDS)

    for index in range(len(manual_details), len(test_cases)):
        case = test_cases[index]
        print(
            f"Answer {case['id']} ({index + 1}/{len(test_cases)})...",
            flush=True,
        )

        manual_result, manual_latency, manual_error = (
            call_pipeline_with_retry(ask_rag, case["question"], limiter)
        )
        manual_detail = evaluate_end_to_end_case(
            case,
            manual_result,
            manual_latency,
            manual_error,
            limiter,
        )

        langchain_result, langchain_latency, langchain_error = (
            call_pipeline_with_retry(
                ask_langchain_full_rag,
                case["question"],
                limiter,
            )
        )
        langchain_detail = evaluate_end_to_end_case(
            case,
            langchain_result,
            langchain_latency,
            langchain_error,
            limiter,
        )

        manual_details.append(manual_detail)
        langchain_details.append(langchain_detail)
        save_benchmark(
            retrieval_comparison,
            manual_details,
            langchain_details,
            completed=False,
        )
        print(
            "  manual: "
            f"answer={manual_detail['answer_correct_deterministic']}, "
            f"faithfulness={manual_detail['judge_faithfulness_score']}; "
            "langchain: "
            f"answer={langchain_detail['answer_correct_deterministic']}, "
            "faithfulness="
            f"{langchain_detail['judge_faithfulness_score']}",
            flush=True,
        )

    output = save_benchmark(
        retrieval_comparison,
        manual_details,
        langchain_details,
        completed=True,
    )
    print_answer_summary(output)
    print(
        "\nResult saved: "
        f"{RESULT_FILE.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
