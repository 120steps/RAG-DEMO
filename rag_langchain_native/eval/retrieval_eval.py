"""Retrieval-only V3 benchmark; never calls answer generation."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

from ..chain import NativeRAGService
from ..config import DEFAULT_SETTINGS
from ..embedding import embedding_config
from ..ingestion import rebuild_knowledge_base
from .common import RESULTS_DIR, expected_rank, load_test_cases, retrieval_summary, write_json


RESULT_FILE = RESULTS_DIR / "retrieval_benchmark.json"


def run_retrieval_evaluation(
    *,
    settings=None,
    result_file=RESULT_FILE,
) -> dict:
    active = settings or replace(DEFAULT_SETTINGS, final_k=10)
    service = NativeRAGService(settings=active)
    details = []
    for case in load_test_cases(answerable_only=True):
        started = time.perf_counter()
        result = service.retrieve_only(case["question"], top_k=10)
        latency = (time.perf_counter() - started) * 1000
        rank = expected_rank(case, result["documents"])
        details.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_source": case.get("expected_source"),
                "expected_page": case.get("expected_page"),
                "hit_rank": rank,
                "latency_ms": latency,
                "original_query": result["original_query"],
                "retrieval_query": result["retrieval_query"],
                "expanded_queries": result["expanded_queries"],
                "documents": result["documents"],
            }
        )
    output = {
        "configuration": {
            "embedding": embedding_config(active),
            "chunk_size": active.chunk_size,
            "chunk_overlap": active.chunk_overlap,
            "collection": active.collection_name,
            "vector_enabled": active.vector_enabled,
            "bm25_enabled": active.bm25_enabled,
            "reranker_enabled": active.reranker_enabled,
            "rewrite_enabled": active.rewrite_enabled,
            "expansion_enabled": active.expansion_enabled,
            "rrf_k": active.rrf_k,
        },
        "summary": retrieval_summary(details),
        "details": details,
    }
    write_json(result_file, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.rebuild:
        print(rebuild_knowledge_base())
    output = run_retrieval_evaluation()
    print("V3 Retrieval Benchmark")
    for k in (1, 3, 5, 10):
        metric = output["summary"][f"hit_at_{k}"]
        print(f"Hit@{k}: {metric['hit_count']}/{output['summary']['total_cases']} ({metric['hit_rate']:.2%})")
    print(f"MRR: {output['summary']['mrr']:.4f}")
    print(f"Average latency: {output['summary']['average_retrieval_latency_ms']:.2f} ms")
    print(f"P95 latency: {output['summary']['p95_retrieval_latency_ms']:.2f} ms")
    print(f"Miss cases: {output['summary']['miss_cases']}")


if __name__ == "__main__":
    main()

