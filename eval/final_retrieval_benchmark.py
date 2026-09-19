import json
import statistics
import sys
import time
from pathlib import Path

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bm25_search import BM25Index
from config import CHUNK_OVERLAP, CHUNK_SIZE
from reranker import RERANKER_MODEL
from retrieve_eval import (
    BM25_RETRIEVAL_K,
    COLLECTION_NAME,
    INITIAL_RETRIEVAL_K,
    load_test_cases,
    retrieve,
    serialize_candidates,
)


CHROMA_DIR = PROJECT_ROOT / "chroma_db"
RESULT_FILE = (
    PROJECT_ROOT
    / "eval"
    / "results"
    / "final_retrieval_benchmark.json"
)
FINAL_TOP_K = 5

MODES = (
    {
        "name": "Vector Only",
        "use_hybrid": False,
        "use_reranker": False,
    },
    {
        "name": "Vector + Reranker",
        "use_hybrid": False,
        "use_reranker": True,
    },
    {
        "name": "Hybrid",
        "use_hybrid": True,
        "use_reranker": False,
    },
    {
        "name": "Hybrid + Reranker",
        "use_hybrid": True,
        "use_reranker": True,
    },
)


def find_hit_rank(candidates, expected_source, expected_page):
    for rank, candidate in enumerate(candidates, start=1):
        metadata = candidate["metadata"]
        if (
            metadata.get("source") == expected_source
            and metadata.get("page") == expected_page
        ):
            return rank
    return None


def warm_up(collection, bm25_index, test_case):
    for mode in MODES:
        retrieve(
            test_case["question"],
            collection,
            use_reranker=mode["use_reranker"],
            use_hybrid=mode["use_hybrid"],
            bm25_index=bm25_index,
            final_top_k=FINAL_TOP_K,
        )


def run_mode(mode, collection, bm25_index, test_cases):
    details = []
    latencies_ms = []
    hit_counts = {1: 0, 3: 0, 5: 0}
    miss_ids = {1: [], 3: [], 5: []}

    for case in test_cases:
        started_at = time.perf_counter()
        candidates = retrieve(
            case["question"],
            collection,
            use_reranker=mode["use_reranker"],
            use_hybrid=mode["use_hybrid"],
            bm25_index=bm25_index,
            final_top_k=FINAL_TOP_K,
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        latencies_ms.append(latency_ms)

        hit_rank = find_hit_rank(
            candidates,
            case["expected_source"],
            case["expected_page"],
        )
        hits = {
            cutoff: hit_rank is not None and hit_rank <= cutoff
            for cutoff in (1, 3, 5)
        }

        for cutoff, hit in hits.items():
            if hit:
                hit_counts[cutoff] += 1
            else:
                miss_ids[cutoff].append(case["id"])

        details.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_source": case["expected_source"],
                "expected_page": case["expected_page"],
                "hit_rank": hit_rank,
                "hit_at_1": hits[1],
                "hit_at_3": hits[3],
                "hit_at_5": hits[5],
                "retrieval_latency_ms": latency_ms,
                "retrieved": serialize_candidates(candidates),
            }
        )

    total_cases = len(test_cases)
    summary = {
        "total_cases": total_cases,
        "hit_at_1_count": hit_counts[1],
        "hit_at_1_rate": hit_counts[1] / total_cases,
        "hit_at_3_count": hit_counts[3],
        "hit_at_3_rate": hit_counts[3] / total_cases,
        "hit_at_5_count": hit_counts[5],
        "hit_at_5_rate": hit_counts[5] / total_cases,
        "miss_count": len(miss_ids[5]),
        "miss_case_ids": miss_ids[5],
        "miss_case_ids_by_cutoff": {
            "hit_at_1": miss_ids[1],
            "hit_at_3": miss_ids[3],
            "hit_at_5": miss_ids[5],
        },
        "average_retrieval_latency_ms": statistics.fmean(
            latencies_ms
        ),
    }
    return {
        "name": mode["name"],
        "parameters": {
            "use_hybrid": mode["use_hybrid"],
            "use_reranker": mode["use_reranker"],
            "vector_top_k": INITIAL_RETRIEVAL_K,
            "bm25_top_k": (
                BM25_RETRIEVAL_K
                if mode["use_hybrid"]
                else None
            ),
            "evaluation_top_k": FINAL_TOP_K,
        },
        "summary": summary,
        "details": details,
    }


def print_summary(results):
    print()
    print("Final Retrieval Benchmark")
    print("=" * 91)
    print(
        f"{'Mode':<22} "
        f"{'Hit@1':>9} "
        f"{'Hit@3':>9} "
        f"{'Hit@5':>9} "
        f"{'Miss@5':>8} "
        f"{'Avg Latency':>15}"
    )
    print("-" * 91)

    for result in results:
        summary = result["summary"]
        print(
            f"{result['name']:<22} "
            f"{summary['hit_at_1_rate']:>8.2%} "
            f"{summary['hit_at_3_rate']:>8.2%} "
            f"{summary['hit_at_5_rate']:>8.2%} "
            f"{summary['miss_count']:>8} "
            f"{summary['average_retrieval_latency_ms']:>12.2f} ms"
        )

    print()
    print("Miss Case IDs by cutoff")
    print("=" * 91)
    for result in results:
        misses = result["summary"]["miss_case_ids_by_cutoff"]
        print(f"{result['name']}:")
        print(f"  Miss@1: {misses['hit_at_1']}")
        print(f"  Miss@3: {misses['hit_at_3']}")
        print(f"  Miss@5: {misses['hit_at_5']}")


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    test_cases = load_test_cases()
    bm25_index = BM25Index.from_chroma(collection)

    print("Warming up local retrieval models...")
    warm_up(collection, bm25_index, test_cases[0])

    results = []
    for mode in MODES:
        print(f"Running: {mode['name']}")
        results.append(
            run_mode(
                mode,
                collection,
                bm25_index,
                test_cases,
            )
        )

    output = {
        "benchmark": {
            "test_case_count": len(test_cases),
            "collection_name": COLLECTION_NAME,
            "collection_count": collection.count(),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "vector_top_k": INITIAL_RETRIEVAL_K,
            "bm25_top_k": BM25_RETRIEVAL_K,
            "evaluation_top_k": FINAL_TOP_K,
            "reranker_model": RERANKER_MODEL,
            "latency_scope": (
                "Warm per-query retrieval: question embedding, candidate "
                "retrieval/merge, and optional reranking. One-time model "
                "loading and BM25 index construction are excluded."
            ),
        },
        "results": results,
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print_summary(results)
    print()
    print(f"Result saved: {RESULT_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
