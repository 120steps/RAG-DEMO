import argparse
import json
import sys
from pathlib import Path

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TOP_K
from bm25_search import BM25Index
from reranker import RERANKER_MODEL
from retrieval import (
    BM25_RETRIEVAL_K,
    VECTOR_RETRIEVAL_K,
    retrieve_candidates,
)


TEST_CASE_FILE = PROJECT_ROOT / "eval" / "test_case.json"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
RESULT_DIR = PROJECT_ROOT / "eval" / "results"
COLLECTION_NAME = "company_knowledge"
INITIAL_RETRIEVAL_K = VECTOR_RETRIEVAL_K
FINAL_TOP_K = TOP_K


def load_test_cases():
    with TEST_CASE_FILE.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    return [
        case
        for case in test_cases
        if case.get("should_answer", True)
        and case.get("expected_source") is not None
        and case.get("expected_page") is not None
    ]


def retrieve(
    question,
    collection,
    use_reranker=False,
    use_hybrid=False,
    bm25_index=None,
    final_top_k=FINAL_TOP_K,
):
    return retrieve_candidates(
        question,
        collection,
        use_reranker=use_reranker,
        use_hybrid=use_hybrid,
        bm25_index=bm25_index,
        final_top_k=final_top_k,
    )


def serialize_candidates(candidates):
    return [
        {
            "rank": rank,
            "document": candidate["document"],
            "metadata": candidate["metadata"],
            "source": candidate["metadata"].get("source"),
            "page": candidate["metadata"].get("page"),
            "chunk_id": candidate["metadata"].get("chunk_id"),
            "original_distance": candidate["original_distance"],
            "bm25_score": candidate["bm25_score"],
            "rerank_score": candidate["rerank_score"],
        }
        for rank, candidate in enumerate(candidates, start=1)
    ]


def evaluate_retrieval(
    use_reranker=False,
    use_hybrid=False,
    final_top_k=FINAL_TOP_K,
    result_file=None,
    verbose=True,
):
    if FINAL_TOP_K != 3:
        raise ValueError(
            f"Reranker experiment requires TOP_K=3; current value is {FINAL_TOP_K}."
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    bm25_index = (
        BM25Index.from_chroma(collection)
        if use_hybrid
        else None
    )
    test_cases = load_test_cases()
    details = []
    top1_hit_count = 0
    hit_at_3_count = 0
    hit_at_5_count = 0

    for case in test_cases:
        candidates = retrieve(
            case["question"],
            collection,
            use_reranker=use_reranker,
            use_hybrid=use_hybrid,
            bm25_index=bm25_index,
            final_top_k=final_top_k,
        )
        expected_source = case["expected_source"]
        expected_page = case["expected_page"]
        hit_rank = None

        for rank, candidate in enumerate(candidates, start=1):
            metadata = candidate["metadata"]
            if (
                metadata.get("source") == expected_source
                and metadata.get("page") == expected_page
            ):
                hit_rank = rank
                break

        top1_hit = hit_rank == 1
        hit_at_3 = hit_rank is not None and hit_rank <= 3
        hit_at_5 = hit_rank is not None and hit_rank <= 5
        if top1_hit:
            top1_hit_count += 1
        if hit_at_3:
            hit_at_3_count += 1
        if hit_at_5:
            hit_at_5_count += 1

        detail = {
            "id": case["id"],
            "question": case["question"],
            "expected_source": expected_source,
            "expected_page": expected_page,
            "top1_hit": top1_hit,
            "hit_at_3": hit_at_3,
            "hit_at_5": hit_at_5,
            "hit_rank": hit_rank,
            "retrieved": serialize_candidates(candidates),
        }
        details.append(detail)

        if verbose:
            print(
                f"{case['id']}: "
                f"Top1={'Hit' if top1_hit else 'Miss'}, "
                f"Hit@3={'Hit' if hit_at_3 else 'Miss'}, "
                f"Hit@5={'Hit' if hit_at_5 else 'Miss'}, "
                f"rank={hit_rank}"
            )

    total_cases = len(test_cases)
    summary = {
        "total_cases": total_cases,
        "top1_hit_count": top1_hit_count,
        "top1_hit_rate": (
            top1_hit_count / total_cases
            if total_cases
            else 0
        ),
        "hit_at_3_count": hit_at_3_count,
        "hit_at_3_rate": (
            hit_at_3_count / total_cases
            if total_cases
            else 0
        ),
        "hit_at_5_count": hit_at_5_count,
        "hit_at_5_rate": (
            hit_at_5_count / total_cases
            if total_cases
            else 0
        ),
    }
    output = {
        "parameters": {
            "use_reranker": use_reranker,
            "use_hybrid": use_hybrid,
            "initial_retrieval_k": INITIAL_RETRIEVAL_K,
            "bm25_retrieval_k": (
                BM25_RETRIEVAL_K
                if use_hybrid
                else None
            ),
            "final_top_k": final_top_k,
            "reranker_model": RERANKER_MODEL if use_reranker else None,
        },
        "summary": summary,
        "details": details,
    }

    if result_file is None:
        result_file = RESULT_DIR / "retrieve_baseline.json"
    else:
        result_file = Path(result_file)

    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")

    if verbose:
        print()
        print("Retrieval Evaluation Summary")
        print("=" * 50)
        print(f"Use Reranker: {use_reranker}")
        print(f"Use Hybrid: {use_hybrid}")
        print(f"Total Cases: {total_cases}")
        print(f"Top1 Hit Rate: {summary['top1_hit_rate']:.2%}")
        print(f"Hit@3: {summary['hit_at_3_rate']:.2%}")
        print(f"Hit@5: {summary['hit_at_5_rate']:.2%}")
        print(f"Result saved: {result_file.relative_to(PROJECT_ROOT)}")

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Rerank Chroma Top-10 before keeping Top-3.",
    )
    parser.add_argument(
        "--use-hybrid",
        action="store_true",
        help="Merge BM25 Top-10 with Vector Top-10.",
    )
    args = parser.parse_args()
    evaluate_retrieval(
        use_reranker=args.use_reranker,
        use_hybrid=args.use_hybrid,
    )


if __name__ == "__main__":
    main()
