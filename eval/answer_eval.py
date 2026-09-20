import json
import re
import sys
import time
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_service import ask_rag


TEST_CASE_FILE = PROJECT_ROOT / "eval" / "test_case.json"
RESULT_FILE = (
    PROJECT_ROOT
    / "eval"
    / "results"
    / "answer_eval_deterministic.json"
)
ANSWER_REQUEST_INTERVAL_SECONDS = 4.0
MAX_REQUEST_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5.0
TEXT_ONLY_BIGRAM_RECALL_THRESHOLD = 0.70
NUMERIC_BIGRAM_RECALL_THRESHOLD = 0.50
REFUSAL_INDICATORS = (
    "无法回答",
    "不能回答",
    "未说明",
    "没有说明",
    "未提供",
    "无法确定",
)


def load_test_cases():
    with TEST_CASE_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text):
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    normalized = normalized.replace("不得", "不")
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalized)


def character_bigram_recall(expected, actual):
    if len(expected) < 2:
        return 1.0 if expected in actual else 0.0

    expected_bigrams = {
        expected[index:index + 2]
        for index in range(len(expected) - 1)
    }
    matched_count = sum(
        bigram in actual
        for bigram in expected_bigrams
    )
    return matched_count / len(expected_bigrams)


def match_expected_content(expected, answer):
    expected_normalized = normalize_text(expected)
    answer_normalized = normalize_text(answer)

    if not expected_normalized:
        return False, "empty_expected_content", 0.0

    if expected_normalized in answer_normalized:
        return True, "normalized_substring", 1.0

    expected_numbers = re.findall(r"\d+(?:\.\d+)?", expected_normalized)
    if any(number not in answer_normalized for number in expected_numbers):
        return False, "missing_number", 0.0

    expected_latin_entities = re.findall(
        r"[a-z][a-z0-9.]*",
        expected_normalized,
    )
    if any(
        entity not in answer_normalized
        for entity in expected_latin_entities
    ):
        return False, "missing_latin_entity", 0.0

    bigram_recall = character_bigram_recall(
        expected_normalized,
        answer_normalized,
    )
    threshold = (
        NUMERIC_BIGRAM_RECALL_THRESHOLD
        if expected_numbers
        else TEXT_ONLY_BIGRAM_RECALL_THRESHOLD
    )
    return (
        bigram_recall >= threshold,
        "content_bigram_recall",
        bigram_recall,
    )


def evaluate_answer_content(answer, expected_answer, expected_keywords):
    answer_normalized = normalize_text(answer)
    if any(
        normalize_text(indicator) in answer_normalized
        for indicator in REFUSAL_INDICATORS
    ):
        return False, "answer_contains_refusal", []

    expected_items = (
        expected_keywords
        if expected_keywords
        else [expected_answer]
    )
    matches = []

    for expected_item in expected_items:
        matched, method, score = match_expected_content(
            expected_item,
            answer,
        )
        matches.append(
            {
                "expected": expected_item,
                "matched": matched,
                "method": method,
                "content_score": score,
            }
        )

    return (
        bool(matches) and all(item["matched"] for item in matches),
        (
            "expected_keywords_content"
            if expected_keywords
            else "expected_answer_content"
        ),
        matches,
    )


def call_rag_with_retry(question, wait_before_request):
    total_latency_ms = 0.0
    last_error = None

    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        if wait_before_request or attempt > 1:
            delay = (
                RETRY_DELAY_SECONDS
                if attempt > 1
                else ANSWER_REQUEST_INTERVAL_SECONDS
            )
            time.sleep(delay)

        started_at = time.perf_counter()
        try:
            result = ask_rag(question)
            total_latency_ms += (
                time.perf_counter() - started_at
            ) * 1000
            return result, total_latency_ms, None
        except Exception as error:
            total_latency_ms += (
                time.perf_counter() - started_at
            ) * 1000
            last_error = str(error)
            print(
                f"  Attempt {attempt}/{MAX_REQUEST_ATTEMPTS} failed: "
                f"{last_error}"
            )

    return None, total_latency_ms, last_error


def build_actual_sources(result):
    if result is None:
        return []

    return [
        {
            "source": metadata.get("source"),
            "page": metadata.get("page"),
            "chunk_id": metadata.get("chunk_id"),
            "distance": distance,
        }
        for metadata, distance in zip(
            result.get("metadatas", []),
            result.get("distances", []),
        )
    ]


def evaluate_case(case, result, latency_ms, evaluation_error):
    should_answer = case["should_answer"]
    answer = result.get("answer", "") if result else ""
    refused = result.get("refused", False) if result else False
    actual_sources = build_actual_sources(result)
    expected_answer = case.get("expected_answer")
    expected_keywords = case.get("expected_keywords", [])
    expected_source = case.get("expected_source")
    expected_page = case.get("expected_page")

    answer_correct = None
    citation_correct = None
    refusal_correct = None
    answer_match_method = None
    expected_content_matches = []
    failure_reasons = []

    if should_answer:
        if evaluation_error or refused:
            answer_correct = False
            answer_match_method = (
                "request_failed"
                if evaluation_error
                else "structured_refusal"
            )
        else:
            (
                answer_correct,
                answer_match_method,
                expected_content_matches,
            ) = evaluate_answer_content(
                answer,
                expected_answer,
                expected_keywords,
            )

        citation_correct = any(
            source["source"] == expected_source
            and source["page"] == expected_page
            for source in actual_sources
        )

        if not answer_correct:
            failure_reasons.append("answer_incorrect")
        if not citation_correct:
            failure_reasons.append("citation_incorrect")
        if refused:
            failure_reasons.append("false_refusal")
    else:
        refusal_correct = bool(result) and refused
        if not refusal_correct:
            failure_reasons.append("refusal_incorrect")

    if evaluation_error:
        failure_reasons.append("request_failed")

    return {
        "id": case["id"],
        "question": case["question"],
        "answer": answer,
        "should_answer": should_answer,
        "refused": refused,
        "answer_correct": answer_correct,
        "answer_match_method": answer_match_method,
        "expected_content_matches": expected_content_matches,
        "citation_correct": citation_correct,
        "refusal_correct": refusal_correct,
        "expected_answer": expected_answer,
        "expected_keywords": expected_keywords,
        "expected_source": expected_source,
        "expected_page": expected_page,
        "actual_sources": actual_sources,
        "latency_ms": latency_ms,
        "failure_reasons": failure_reasons,
        "evaluation_error": evaluation_error,
    }


def build_summary(details):
    answerable_details = [
        detail
        for detail in details
        if detail["should_answer"]
    ]
    refusal_details = [
        detail
        for detail in details
        if not detail["should_answer"]
    ]
    answered_cases = len(answerable_details)
    refusal_cases = len(refusal_details)
    answer_correct_count = sum(
        detail["answer_correct"] is True
        for detail in answerable_details
    )
    citation_correct_count = sum(
        detail["citation_correct"] is True
        for detail in answerable_details
    )
    refusal_correct_count = sum(
        detail["refusal_correct"] is True
        for detail in refusal_details
    )
    failed_case_ids = [
        detail["id"]
        for detail in details
        if detail["failure_reasons"]
    ]

    return {
        "total_cases": len(details),
        "answered_cases": answered_cases,
        "refusal_cases": refusal_cases,
        "answer_correct_count": answer_correct_count,
        "answer_accuracy": (
            answer_correct_count / answered_cases
            if answered_cases
            else 0
        ),
        "citation_correct_count": citation_correct_count,
        "citation_accuracy": (
            citation_correct_count / answered_cases
            if answered_cases
            else 0
        ),
        "refusal_correct_count": refusal_correct_count,
        "refusal_accuracy": (
            refusal_correct_count / refusal_cases
            if refusal_cases
            else 0
        ),
        "failed_case_ids": failed_case_ids,
    }


def main():
    test_cases = load_test_cases()
    details = []

    for index, case in enumerate(test_cases):
        print(
            f"Running {case['id']} "
            f"({index + 1}/{len(test_cases)})..."
        )
        result, latency_ms, evaluation_error = call_rag_with_retry(
            case["question"],
            wait_before_request=index > 0,
        )
        detail = evaluate_case(
            case,
            result,
            latency_ms,
            evaluation_error,
        )
        details.append(detail)
        print(
            f"  answer_correct={detail['answer_correct']}, "
            f"citation_correct={detail['citation_correct']}, "
            f"refusal_correct={detail['refusal_correct']}, "
            f"latency={latency_ms:.2f} ms"
        )

    summary = build_summary(details)
    output = {
        "parameters": {
            "test_case_file": str(
                TEST_CASE_FILE.relative_to(PROJECT_ROOT)
            ),
            "answer_request_interval_seconds": (
                ANSWER_REQUEST_INTERVAL_SECONDS
            ),
            "max_request_attempts": MAX_REQUEST_ATTEMPTS,
            "evaluation_type": "deterministic",
            "llm_as_judge": False,
        },
        "summary": summary,
        "details": details,
    }

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print()
    print("Deterministic Answer Evaluation")
    print("=" * 60)
    print(f"Total Cases: {summary['total_cases']}")
    print(f"Answered Cases: {summary['answered_cases']}")
    print(f"Refusal Cases: {summary['refusal_cases']}")
    print(
        f"Answer Accuracy: "
        f"{summary['answer_correct_count']}/"
        f"{summary['answered_cases']} "
        f"({summary['answer_accuracy']:.2%})"
    )
    print(
        f"Citation Accuracy: "
        f"{summary['citation_correct_count']}/"
        f"{summary['answered_cases']} "
        f"({summary['citation_accuracy']:.2%})"
    )
    print(
        f"Refusal Accuracy: "
        f"{summary['refusal_correct_count']}/"
        f"{summary['refusal_cases']} "
        f"({summary['refusal_accuracy']:.2%})"
    )
    print(f"Failed Case IDs: {summary['failed_case_ids']}")
    print(
        f"Result saved: {RESULT_FILE.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
