"""Phase 9 permission-aware Retrieval Benchmark；不调用 Answer Generation。"""

from __future__ import annotations

import argparse
import time

from ..config import DEFAULT_SETTINGS
from ..enterprise import EnterpriseRAGService
from .common import RESULTS_DIR, expected_rank, load_test_cases, retrieval_summary, write_json
from .enterprise_common import (
    EVAL_KNOWLEDGE_BASE,
    bootstrap_evaluation_knowledge_base,
    evaluation_principal,
)


RESULT_FILE = RESULTS_DIR / "phase9_retrieval_benchmark.json"


def run(*, bootstrap: bool = False) -> dict:
    if bootstrap:
        state = bootstrap_evaluation_knowledge_base(DEFAULT_SETTINGS)
        catalog = state["catalog"]
        principal = state["principal"]
    else:
        from ..catalog import DocumentCatalog

        catalog = DocumentCatalog(DEFAULT_SETTINGS.catalog_path)
        principal = evaluation_principal(catalog)
    service = EnterpriseRAGService(catalog, DEFAULT_SETTINGS)
    details = []
    for case in load_test_cases(answerable_only=True):
        started = time.perf_counter()
        result = service.retrieve_only(
            principal=principal,
            knowledge_base_id=EVAL_KNOWLEDGE_BASE,
            question=case["question"],
            top_k=10,
        )
        elapsed = (time.perf_counter() - started) * 1000
        details.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_source": case.get("expected_source"),
                "expected_page": case.get("expected_page"),
                "hit_rank": expected_rank(case, result["documents"]),
                "latency_ms": elapsed,
                "tenant_id": result["tenant_id"],
                "authorization_epoch": result["authorization_epoch"],
                "documents": result["documents"],
            }
        )
    summary = retrieval_summary(details)
    output = {
        "configuration": {
            "tenant_id": principal.tenant_id,
            "roles": list(principal.roles),
            "knowledge_base_id": EVAL_KNOWLEDGE_BASE,
            "collection": DEFAULT_SETTINGS.enterprise_collection_name,
            "chunk_size": DEFAULT_SETTINGS.chunk_size,
            "chunk_overlap": DEFAULT_SETTINGS.chunk_overlap,
            "embedding_model": DEFAULT_SETTINGS.embedding_model,
            "vector_enabled": DEFAULT_SETTINGS.vector_enabled,
            "bm25_enabled": DEFAULT_SETTINGS.bm25_enabled,
            "reranker_enabled": DEFAULT_SETTINGS.reranker_enabled,
        },
        "summary": summary,
        "details": details,
    }
    write_json(RESULT_FILE, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    output = run(bootstrap=args.bootstrap)
    summary = output["summary"]
    print("Phase 9 Authorized Retrieval Benchmark")
    for k in (1, 3, 5, 10):
        value = summary[f"hit_at_{k}"]
        print(f"Hit@{k}: {value['hit_count']}/{summary['total_cases']} ({value['hit_rate']:.2%})")
    print(f"MRR: {summary['mrr']:.4f}")
    print(f"Average latency: {summary['average_retrieval_latency_ms']:.2f} ms")
    print(f"P95 latency: {summary['p95_retrieval_latency_ms']:.2f} ms")
    print(f"Miss cases: {summary['miss_cases']}")


if __name__ == "__main__":
    main()

