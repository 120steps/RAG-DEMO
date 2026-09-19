import json
import sys
import time
from pathlib import Path

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bm25_search import BM25Index
from config import CHUNK_OVERLAP, CHUNK_SIZE
from query_rewriter import CACHE_FILE, rewrite_query
from reranker import RERANKER_MODEL
from retrieve_eval import (
    BM25_RETRIEVAL_K,
    CHROMA_DIR,
    COLLECTION_NAME,
    INITIAL_RETRIEVAL_K,
    load_test_cases,
    retrieve,
    serialize_candidates,
)


RESULT_DIR = PROJECT_ROOT / "eval" / "results"
OFF_RESULT_FILE = RESULT_DIR / "query_rewrite_off.json"
ON_RESULT_FILE = RESULT_DIR / "query_rewrite_on.json"
EVALUATION_TOP_K = 10
HIT_CUTOFFS = (1, 3, 5)
GEMINI_CALL_INTERVAL_SECONDS = 3.5


def load_rewrite_cache():
    try:
        with CACHE_FILE.open("r", encoding="utf-8") as file:
            cache = json.load(file)
        return cache if isinstance(cache, dict) else {}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def prepare_rewritten_queries(test_cases):
    cache = load_rewrite_cache()
    rewritten_queries = {}
    cache_hits = {}
    last_gemini_call_started = None

    for case in test_cases:
        case_id = case["id"]
        original_query = case["question"]
        cached_query = cache.get(original_query)

        if isinstance(cached_query, str) and cached_query:
            rewritten_queries[case_id] = cached_query
            cache_hits[case_id] = True
            print(f"Rewrite {case_id}: cache hit")
            continue

        if last_gemini_call_started is not None:
            elapsed = time.monotonic() - last_gemini_call_started
            remaining = GEMINI_CALL_INTERVAL_SECONDS - elapsed
            if remaining > 0:
                time.sleep(remaining)

        print(f"Rewrite {case_id}: calling Gemini")
        last_gemini_call_started = time.monotonic()
        rewritten_query = rewrite_query(original_query)
        rewritten_queries[case_id] = rewritten_query
        cache_hits[case_id] = False

        cache = load_rewrite_cache()

    return rewritten_queries, cache_hits


def find_hit_rank(candidates, expected_source, expected_page):
    for rank, candidate in enumerate(candidates, start=1):
        metadata = candidate["metadata"]
        if (
            metadata.get("source") == expected_source
            and metadata.get("page") == expected_page
        ):
            return rank
    return None


def retrieve_cases(test_cases, queries, collection, bm25_index, label):
    results = {}

    for case in test_cases:
        case_id = case["id"]
        print(f"{label} {case_id}")
        candidates = retrieve(
            queries[case_id],
            collection,
            use_hybrid=True,
            use_reranker=True,
            bm25_index=bm25_index,
            final_top_k=EVALUATION_TOP_K,
        )
        results[case_id] = {
            "hit_rank": find_hit_rank(
                candidates,
                case["expected_source"],
                case["expected_page"],
            ),
            "retrieved": serialize_candidates(candidates),
        }

    return results


def build_summary(test_cases, results):
    total_cases = len(test_cases)
    summary = {"total_cases": total_cases}

    for cutoff in HIT_CUTOFFS:
        hit_count = sum(
            result["hit_rank"] is not None
            and result["hit_rank"] <= cutoff
            for result in results.values()
        )
        summary[f"hit_at_{cutoff}_count"] = hit_count
        summary[f"hit_at_{cutoff}_rate"] = (
            hit_count / total_cases
            if total_cases
            else 0
        )

    return summary


def classify_case(off_rank, on_rank):
    if off_rank == on_rank:
        return "unchanged"
    if off_rank is None:
        return "improved"
    if on_rank is None:
        return "degraded"
    return "improved" if on_rank < off_rank else "degraded"


def build_details(
    test_cases,
    rewritten_queries,
    cache_hits,
    off_results,
    on_results,
    result_key,
):
    details = []

    for case in test_cases:
        case_id = case["id"]
        off_rank = off_results[case_id]["hit_rank"]
        on_rank = on_results[case_id]["hit_rank"]
        details.append(
            {
                "id": case_id,
                "original_query": case["question"],
                "rewritten_query": rewritten_queries[case_id],
                "expected_source": case["expected_source"],
                "expected_page": case["expected_page"],
                "hit_rank_without_rewrite": off_rank,
                "hit_rank_with_rewrite": on_rank,
                "comparison": classify_case(off_rank, on_rank),
                "rewrite_cache_hit": cache_hits[case_id],
                "retrieved": (
                    off_results[case_id]["retrieved"]
                    if result_key == "off"
                    else on_results[case_id]["retrieved"]
                ),
            }
        )

    return details


def write_result(path, enabled, summary, details):
    output = {
        "parameters": {
            "query_rewrite_enabled": enabled,
            "use_hybrid": True,
            "use_reranker": True,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "vector_top_k": INITIAL_RETRIEVAL_K,
            "bm25_top_k": BM25_RETRIEVAL_K,
            "evaluation_top_k": EVALUATION_TOP_K,
            "reranker_model": RERANKER_MODEL,
        },
        "summary": summary,
        "details": details,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")


def format_rank(rank):
    return str(rank) if rank is not None else "Miss"


def print_summary(off_summary, on_summary):
    print()
    print("Query Rewrite Retrieval A/B Test")
    print("=" * 63)
    print(f"{'Mode':<20} {'Hit@1':>12} {'Hit@3':>12} {'Hit@5':>12}")
    print("-" * 63)

    for name, summary in (
        ("Rewrite OFF", off_summary),
        ("Rewrite ON", on_summary),
    ):
        print(
            f"{name:<20} "
            f"{summary['hit_at_1_rate']:>11.2%} "
            f"{summary['hit_at_3_rate']:>11.2%} "
            f"{summary['hit_at_5_rate']:>11.2%}"
        )


def print_comparisons(test_cases, off_results, on_results):
    categories = {
        "improved": [],
        "unchanged": [],
        "degraded": [],
    }

    for case in test_cases:
        case_id = case["id"]
        off_rank = off_results[case_id]["hit_rank"]
        on_rank = on_results[case_id]["hit_rank"]
        category = classify_case(off_rank, on_rank)
        categories[category].append((case_id, off_rank, on_rank))

    for category, title in (
        ("improved", "Improved Cases"),
        ("unchanged", "Unchanged Cases"),
        ("degraded", "Degraded Cases"),
    ):
        print()
        print(f"{title} ({len(categories[category])})")
        print("-" * 63)
        if not categories[category]:
            print("None")
            continue

        for case_id, off_rank, on_rank in categories[category]:
            print(
                f"{case_id}: OFF rank = {format_rank(off_rank)}, "
                f"ON rank = {format_rank(on_rank)}"
            )


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    bm25_index = BM25Index.from_chroma(collection)
    test_cases = load_test_cases()

    original_queries = {
        case["id"]: case["question"]
        for case in test_cases
    }

    print("Running Rewrite OFF retrieval...")
    off_results = retrieve_cases(
        test_cases,
        original_queries,
        collection,
        bm25_index,
        "OFF",
    )

    print()
    print("Preparing rewritten queries...")
    rewritten_queries, cache_hits = prepare_rewritten_queries(test_cases)

    print()
    print("Running Rewrite ON retrieval...")
    on_results = retrieve_cases(
        test_cases,
        rewritten_queries,
        collection,
        bm25_index,
        "ON",
    )

    off_summary = build_summary(test_cases, off_results)
    on_summary = build_summary(test_cases, on_results)
    off_details = build_details(
        test_cases,
        rewritten_queries,
        cache_hits,
        off_results,
        on_results,
        "off",
    )
    on_details = build_details(
        test_cases,
        rewritten_queries,
        cache_hits,
        off_results,
        on_results,
        "on",
    )

    write_result(
        OFF_RESULT_FILE,
        False,
        off_summary,
        off_details,
    )
    write_result(
        ON_RESULT_FILE,
        True,
        on_summary,
        on_details,
    )

    print_summary(off_summary, on_summary)
    print_comparisons(test_cases, off_results, on_results)
    print()
    print(
        f"Result saved: {OFF_RESULT_FILE.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Result saved: {ON_RESULT_FILE.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
