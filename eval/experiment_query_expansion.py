import json
import math
import statistics
import sys
import time
from pathlib import Path

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bm25_search import BM25Index
from config import CHUNK_OVERLAP, CHUNK_SIZE, QUERY_EXPANSION_COUNT
from query_expander import (
    CACHE_FILE as EXPANSION_CACHE_FILE,
    expand_query_with_status,
    get_cached_expansion,
)
from query_rewriter import (
    CACHE_FILE as REWRITE_CACHE_FILE,
    rewrite_query,
)
from reranker import RERANKER_MODEL
from result_fusion import RRF_K
from retrieve_eval import (
    BM25_RETRIEVAL_K,
    CHROMA_DIR,
    COLLECTION_NAME,
    INITIAL_RETRIEVAL_K,
    load_test_cases,
    retrieve,
    serialize_candidates,
)
from retrieval import RERANK_CANDIDATE_K


RESULT_DIR = PROJECT_ROOT / "eval" / "results"
SUMMARY_FILE = RESULT_DIR / "query_expansion_summary.json"
EVALUATION_TOP_K = 10
HIT_CUTOFFS = (1, 3, 5, 10)
GEMINI_CALL_INTERVAL_SECONDS = 3.5

MODES = (
    {
        "key": "base",
        "name": "Base Retrieval",
        "use_rewrite": False,
        "use_expansion": False,
        "result_file": RESULT_DIR / "query_expansion_base.json",
        "comparison_reference": None,
    },
    {
        "key": "rewrite",
        "name": "Rewrite Only",
        "use_rewrite": True,
        "use_expansion": False,
        "result_file": RESULT_DIR / "query_expansion_rewrite.json",
        "comparison_reference": "base",
    },
    {
        "key": "expansion",
        "name": "Expansion Only",
        "use_rewrite": False,
        "use_expansion": True,
        "result_file": RESULT_DIR / "query_expansion_only.json",
        "comparison_reference": "base",
    },
    {
        "key": "rewrite_expansion",
        "name": "Rewrite + Expansion",
        "use_rewrite": True,
        "use_expansion": True,
        "result_file": RESULT_DIR / "query_expansion_rewrite_plus.json",
        "comparison_reference": "rewrite",
    },
)


class GeminiCallLimiter:
    def __init__(self, interval_seconds):
        self.interval_seconds = interval_seconds
        self.last_call_started = None

    def wait(self):
        if self.last_call_started is not None:
            elapsed = time.monotonic() - self.last_call_started
            remaining = self.interval_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self.last_call_started = time.monotonic()


def load_json_cache(path):
    try:
        with path.open("r", encoding="utf-8") as file:
            cache = json.load(file)
        return cache if isinstance(cache, dict) else {}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def prepare_rewrites(test_cases, limiter):
    cache = load_json_cache(REWRITE_CACHE_FILE)
    rewrites = {}

    for case in test_cases:
        query = case["question"]
        cached_query = cache.get(query)
        if isinstance(cached_query, str) and cached_query:
            rewrites[case["id"]] = cached_query
            print(f"Rewrite {case['id']}: cache hit")
            continue

        limiter.wait()
        print(f"Rewrite {case['id']}: calling Gemini")
        rewrites[case["id"]] = rewrite_query(query)
        cache = load_json_cache(REWRITE_CACHE_FILE)

    return rewrites


def prepare_expansions(queries, limiter):
    expansions = {}
    failures = {}
    cache_hits = {}

    for label, query in queries:
        if query in expansions:
            continue

        cached_queries = get_cached_expansion(query)
        if cached_queries is not None:
            expansions[query] = cached_queries
            failures[query] = False
            cache_hits[query] = True
            print(f"Expansion {label}: cache hit")
            continue

        limiter.wait()
        print(f"Expansion {label}: calling Gemini")
        expanded_queries, failed = expand_query_with_status(query)
        expansions[query] = expanded_queries
        failures[query] = failed
        cache_hits[query] = False

    return expansions, failures, cache_hits


def find_hit_rank(candidates, expected_source, expected_page):
    for rank, candidate in enumerate(candidates, start=1):
        metadata = candidate["metadata"]
        if (
            metadata.get("source") == expected_source
            and metadata.get("page") == expected_page
        ):
            return rank
    return None


def percentile_95(values):
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def classify_rank(reference_rank, current_rank):
    if reference_rank == current_rank:
        return "unchanged"
    if reference_rank is None:
        return "improved"
    if current_rank is None:
        return "degraded"
    return "improved" if current_rank < reference_rank else "degraded"


def run_mode(
    mode,
    test_cases,
    rewrites,
    expansions,
    collection,
    bm25_index,
):
    case_results = {}
    latencies_ms = []

    for case in test_cases:
        case_id = case["id"]
        original_query = case["question"]
        retrieval_query = (
            rewrites[case_id]
            if mode["use_rewrite"]
            else original_query
        )
        expanded_queries = (
            expansions[retrieval_query]
            if mode["use_expansion"]
            else None
        )

        started_at = time.perf_counter()
        candidates = retrieve(
            retrieval_query,
            collection,
            use_hybrid=True,
            use_reranker=True,
            bm25_index=bm25_index,
            final_top_k=EVALUATION_TOP_K,
            expanded_queries=expanded_queries,
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        latencies_ms.append(latency_ms)

        hit_rank = find_hit_rank(
            candidates,
            case["expected_source"],
            case["expected_page"],
        )
        case_results[case_id] = {
            "hit_rank": hit_rank,
            "latency_ms": latency_ms,
            "retrieval_query": retrieval_query,
            "expanded_queries": (
                expanded_queries
                if expanded_queries is not None
                else [retrieval_query]
            ),
            "retrieved": serialize_candidates(candidates),
        }
        print(
            f"{mode['name']} {case_id}: "
            f"rank={hit_rank}, latency={latency_ms:.2f} ms"
        )

    total_cases = len(test_cases)
    summary = {
        "mode": mode["name"],
        "total_cases": total_cases,
    }
    for cutoff in HIT_CUTOFFS:
        count = sum(
            result["hit_rank"] is not None
            and result["hit_rank"] <= cutoff
            for result in case_results.values()
        )
        summary[f"hit_at_{cutoff}"] = (
            count / total_cases if total_cases else 0
        )
        summary[f"hit_at_{cutoff}_count"] = count

    reciprocal_ranks = [
        1 / result["hit_rank"]
        if result["hit_rank"] is not None
        else 0
        for result in case_results.values()
    ]
    summary["mrr"] = (
        statistics.fmean(reciprocal_ranks)
        if reciprocal_ranks
        else 0
    )
    summary["average_latency_ms"] = (
        statistics.fmean(latencies_ms)
        if latencies_ms
        else 0
    )
    summary["p95_latency_ms"] = percentile_95(latencies_ms)
    summary["miss_cases"] = [
        case_id
        for case_id, result in case_results.items()
        if result["hit_rank"] is None
    ]

    return case_results, summary


def add_comparisons(mode_results, mode_summaries, test_cases):
    for mode in MODES:
        reference_key = mode["comparison_reference"]
        improved = []
        unchanged = []
        degraded = []

        for case in test_cases:
            case_id = case["id"]
            current_rank = mode_results[mode["key"]][case_id]["hit_rank"]
            reference_rank = (
                current_rank
                if reference_key is None
                else mode_results[reference_key][case_id]["hit_rank"]
            )
            classification = classify_rank(reference_rank, current_rank)
            if classification == "improved":
                improved.append(case_id)
            elif classification == "degraded":
                degraded.append(case_id)
            else:
                unchanged.append(case_id)

        summary = mode_summaries[mode["key"]]
        summary["comparison_reference"] = reference_key
        summary["improved_cases"] = improved
        summary["unchanged_cases"] = unchanged
        summary["degraded_cases"] = degraded


def build_details(
    mode,
    test_cases,
    rewrites,
    expansions,
    expansion_failures,
    mode_results,
):
    details = []

    for case in test_cases:
        case_id = case["id"]
        original_query = case["question"]
        rewritten_query = rewrites[case_id]
        result = mode_results[mode["key"]][case_id]
        reference_key = mode["comparison_reference"]
        reference_rank = (
            result["hit_rank"]
            if reference_key is None
            else mode_results[reference_key][case_id]["hit_rank"]
        )
        expansion_input = (
            rewritten_query
            if mode["use_rewrite"]
            else original_query
        )

        details.append(
            {
                "id": case_id,
                "question": original_query,
                "expected_source": case["expected_source"],
                "expected_page": case["expected_page"],
                "original_query": original_query,
                "rewritten_query": rewritten_query,
                "expanded_queries": result["expanded_queries"],
                "base_rank": mode_results["base"][case_id]["hit_rank"],
                "rewrite_rank": mode_results["rewrite"][case_id]["hit_rank"],
                "expansion_rank": mode_results["expansion"][case_id]["hit_rank"],
                "rewrite_expansion_rank": mode_results[
                    "rewrite_expansion"
                ][case_id]["hit_rank"],
                "comparison": classify_rank(
                    reference_rank,
                    result["hit_rank"],
                ),
                "expansion_failed": (
                    expansion_failures[expansion_input]
                    if mode["use_expansion"]
                    else False
                ),
                "retrieval_latency_ms": result["latency_ms"],
                "retrieved": result["retrieved"],
            }
        )

    return details


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)
        file.write("\n")


def print_summary(mode_summaries):
    print()
    print("Query Expansion Retrieval Benchmark")
    print("=" * 110)
    print(
        f"{'Mode':<24} "
        f"{'Hit@1':>9} {'Hit@3':>9} {'Hit@5':>9} {'Hit@10':>9} "
        f"{'Avg ms':>12} {'P95 ms':>12}"
    )
    print("-" * 110)

    for mode in MODES:
        summary = mode_summaries[mode["key"]]
        print(
            f"{mode['name']:<24} "
            f"{summary['hit_at_1']:>8.2%} "
            f"{summary['hit_at_3']:>8.2%} "
            f"{summary['hit_at_5']:>8.2%} "
            f"{summary['hit_at_10']:>8.2%} "
            f"{summary['average_latency_ms']:>11.2f} "
            f"{summary['p95_latency_ms']:>11.2f}"
        )

    for mode in MODES:
        summary = mode_summaries[mode["key"]]
        print()
        print(f"{mode['name']}:")
        print(f"  Improved Cases: {summary['improved_cases']}")
        print(f"  Degraded Cases: {summary['degraded_cases']}")
        print(f"  Top10 Miss Cases: {summary['miss_cases']}")


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    bm25_index = BM25Index.from_chroma(collection)
    test_cases = load_test_cases()
    limiter = GeminiCallLimiter(GEMINI_CALL_INTERVAL_SECONDS)

    print("Preparing rewritten queries...")
    rewrites = prepare_rewrites(test_cases, limiter)

    expansion_inputs = []
    for case in test_cases:
        expansion_inputs.append((f"{case['id']} original", case["question"]))
        expansion_inputs.append((f"{case['id']} rewritten", rewrites[case["id"]]))

    print()
    print("Preparing expanded queries...")
    expansions, expansion_failures, expansion_cache_hits = (
        prepare_expansions(expansion_inputs, limiter)
    )

    print()
    print("Warming up local retrieval models...")
    first_case = test_cases[0]
    retrieve(
        first_case["question"],
        collection,
        use_hybrid=True,
        use_reranker=True,
        bm25_index=bm25_index,
        final_top_k=EVALUATION_TOP_K,
    )

    mode_results = {}
    mode_summaries = {}
    for mode in MODES:
        print()
        print(f"Running {mode['name']}...")
        results, summary = run_mode(
            mode,
            test_cases,
            rewrites,
            expansions,
            collection,
            bm25_index,
        )
        mode_results[mode["key"]] = results
        mode_summaries[mode["key"]] = summary

    add_comparisons(mode_results, mode_summaries, test_cases)

    common_parameters = {
        "test_case_count": len(test_cases),
        "collection_name": COLLECTION_NAME,
        "collection_count": collection.count(),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "use_hybrid": True,
        "use_reranker": True,
        "vector_top_k": INITIAL_RETRIEVAL_K,
        "bm25_top_k": BM25_RETRIEVAL_K,
        "rerank_candidate_k": RERANK_CANDIDATE_K,
        "evaluation_top_k": EVALUATION_TOP_K,
        "query_expansion_count": QUERY_EXPANSION_COUNT,
        "rrf_k": RRF_K,
        "reranker_model": RERANKER_MODEL,
        "latency_scope": (
            "Embedding, Vector Search, BM25, optional RRF fusion, and "
            "one final rerank. Cached/Gemini query preparation is excluded."
        ),
    }

    for mode in MODES:
        details = build_details(
            mode,
            test_cases,
            rewrites,
            expansions,
            expansion_failures,
            mode_results,
        )
        write_json(
            mode["result_file"],
            {
                "mode": mode["name"],
                "parameters": {
                    **common_parameters,
                    "query_rewrite_enabled": mode["use_rewrite"],
                    "query_expansion_enabled": mode["use_expansion"],
                },
                "summary": mode_summaries[mode["key"]],
                "details": details,
            },
        )

    summary_output = {
        "parameters": common_parameters,
        "query_preparation": {
            "gemini_call_interval_seconds": GEMINI_CALL_INTERVAL_SECONDS,
            "expansion_cache_file": str(
                EXPANSION_CACHE_FILE.relative_to(PROJECT_ROOT)
            ),
            "expansion_cache_hit_count": sum(expansion_cache_hits.values()),
            "expansion_failed_queries": [
                query
                for query, failed in expansion_failures.items()
                if failed
            ],
        },
        "modes": [
            mode_summaries[mode["key"]]
            for mode in MODES
        ],
    }
    write_json(SUMMARY_FILE, summary_output)

    print_summary(mode_summaries)
    print()
    for mode in MODES:
        print(
            f"Result saved: "
            f"{mode['result_file'].relative_to(PROJECT_ROOT)}"
        )
    print(f"Summary saved: {SUMMARY_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
