import json
import sys
from pathlib import Path
from typing import Any

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedding import embed_text
from langchain_rag.lc_embedding import ExistingEmbeddingAdapter
from langchain_rag.lc_vectorstore import (
    DISTANCE_SPACE,
    LANGCHAIN_CHROMA_DIR,
    LANGCHAIN_COLLECTION_NAME,
    SOURCE_CHROMA_DIR,
    SOURCE_COLLECTION_NAME,
    initialize_vectorstore,
)


TEST_CASE_FILE = PROJECT_ROOT / "eval" / "test_case.json"
RESULT_FILE = (
    PROJECT_ROOT
    / "eval"
    / "results"
    / "langchain_vector_comparison.json"
)
TOP_K = 10
HIT_K_VALUES = (1, 3, 5, 10)
DISTANCE_TOLERANCE = 1e-6
EMBEDDING_TOLERANCE = 1e-7


def load_test_cases() -> list[dict[str, Any]]:
    with TEST_CASE_FILE.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    return [
        case
        for case in test_cases
        if case.get("should_answer", True)
        and case.get("expected_source") is not None
        and case.get("expected_page") is not None
    ]


def query_manual_vector(
    collection,
    question: str,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[float]]:
    query_embedding = embed_text(question)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    candidates = [
        {
            "id": document_id,
            "document": document,
            "metadata": metadata,
            "distance": float(distance),
        }
        for document_id, document, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]
    return candidates, query_embedding


def query_langchain_vector(
    vectorstore,
    question: str,
    top_k: int,
) -> list[dict[str, Any]]:
    results = vectorstore.similarity_search_with_score(
        question,
        k=top_k,
    )
    return [
        {
            "id": document.id,
            "document": document.page_content,
            "metadata": dict(document.metadata),
            "distance": float(distance),
        }
        for document, distance in results
    ]


def serialize_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "id": candidate["id"],
            "source": candidate["metadata"].get("source"),
            "page": candidate["metadata"].get("page"),
            "chunk_id": candidate["metadata"].get("chunk_id"),
            "distance": candidate["distance"],
            "metadata": candidate["metadata"],
            "document": candidate["document"],
        }
        for rank, candidate in enumerate(candidates, start=1)
    ]


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


def compare_query_embeddings(
    manual_embedding: list[float],
    langchain_embedding: list[float],
) -> dict[str, Any]:
    dimension_match = len(manual_embedding) == len(langchain_embedding)
    differences = [
        abs(manual_value - langchain_value)
        for manual_value, langchain_value in zip(
            manual_embedding,
            langchain_embedding,
        )
    ]
    max_difference = max(differences, default=0.0)
    return {
        "manual_dimension": len(manual_embedding),
        "langchain_dimension": len(langchain_embedding),
        "dimension_match": dimension_match,
        "max_absolute_difference": max_difference,
        "match": (
            dimension_match
            and max_difference <= EMBEDDING_TOLERANCE
        ),
    }


def compare_candidates(
    manual_candidates: list[dict[str, Any]],
    langchain_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    manual_ids = [candidate["id"] for candidate in manual_candidates]
    langchain_ids = [
        candidate["id"] for candidate in langchain_candidates
    ]
    order_differences = [
        {
            "rank": rank,
            "manual_id": manual_id,
            "langchain_id": langchain_id,
        }
        for rank, (manual_id, langchain_id) in enumerate(
            zip(manual_ids, langchain_ids),
            start=1,
        )
        if manual_id != langchain_id
    ]

    manual_by_id = {
        candidate["id"]: candidate for candidate in manual_candidates
    }
    langchain_by_id = {
        candidate["id"]: candidate for candidate in langchain_candidates
    }
    id_sets_match = set(manual_by_id) == set(langchain_by_id)
    common_ids = set(manual_by_id) & set(langchain_by_id)

    metadata_mismatches = [
        document_id
        for document_id in sorted(common_ids)
        if (
            manual_by_id[document_id]["metadata"]
            != langchain_by_id[document_id]["metadata"]
        )
    ]
    content_mismatches = [
        document_id
        for document_id in sorted(common_ids)
        if (
            manual_by_id[document_id]["document"]
            != langchain_by_id[document_id]["document"]
        )
    ]
    distance_differences = {
        document_id: abs(
            manual_by_id[document_id]["distance"]
            - langchain_by_id[document_id]["distance"]
        )
        for document_id in sorted(common_ids)
    }
    max_distance_difference = max(
        distance_differences.values(),
        default=0.0,
    )

    return {
        "top_k_id_order_match": manual_ids == langchain_ids,
        "id_sets_match": id_sets_match,
        "id_order_differences": order_differences,
        "manual_only_ids": sorted(set(manual_by_id) - set(langchain_by_id)),
        "langchain_only_ids": sorted(
            set(langchain_by_id) - set(manual_by_id)
        ),
        "metadata_match": id_sets_match and not metadata_mismatches,
        "metadata_mismatch_ids": metadata_mismatches,
        "chunk_content_match": id_sets_match and not content_mismatches,
        "content_mismatch_ids": content_mismatches,
        "max_vector_distance_difference": max_distance_difference,
        "vector_distances_match": (
            id_sets_match
            and max_distance_difference <= DISTANCE_TOLERANCE
        ),
    }


def compare_collections(
    manual_collection,
    vectorstore,
) -> dict[str, Any]:
    manual_data = manual_collection.get(
        include=["documents", "metadatas"],
    )
    langchain_data = vectorstore.get(
        include=["documents", "metadatas"],
    )

    manual_by_id = {
        document_id: (document, metadata)
        for document_id, document, metadata in zip(
            manual_data["ids"],
            manual_data["documents"],
            manual_data["metadatas"],
        )
    }
    langchain_by_id = {
        document_id: (document, metadata)
        for document_id, document, metadata in zip(
            langchain_data["ids"],
            langchain_data["documents"],
            langchain_data["metadatas"],
        )
    }
    ids_match = set(manual_by_id) == set(langchain_by_id)
    common_ids = set(manual_by_id) & set(langchain_by_id)
    content_mismatches = [
        document_id
        for document_id in sorted(common_ids)
        if manual_by_id[document_id][0] != langchain_by_id[document_id][0]
    ]
    metadata_mismatches = [
        document_id
        for document_id in sorted(common_ids)
        if manual_by_id[document_id][1] != langchain_by_id[document_id][1]
    ]

    return {
        "manual_chunk_count": len(manual_by_id),
        "langchain_chunk_count": len(langchain_by_id),
        "stable_ids_match": ids_match,
        "chunk_content_match": ids_match and not content_mismatches,
        "metadata_match": ids_match and not metadata_mismatches,
        "manual_only_ids": sorted(set(manual_by_id) - set(langchain_by_id)),
        "langchain_only_ids": sorted(
            set(langchain_by_id) - set(manual_by_id)
        ),
        "content_mismatch_ids": content_mismatches,
        "metadata_mismatch_ids": metadata_mismatches,
    }


def build_metrics(
    hit_ranks: list[int | None],
) -> dict[str, Any]:
    total_cases = len(hit_ranks)
    metrics = {"total_cases": total_cases}
    for top_k in HIT_K_VALUES:
        hit_count = sum(
            rank is not None and rank <= top_k
            for rank in hit_ranks
        )
        metrics[f"hit_at_{top_k}"] = {
            "hit_count": hit_count,
            "hit_rate": hit_count / total_cases if total_cases else 0.0,
        }
    return metrics


def get_collection_space(directory: Path, collection_name: str) -> str:
    client = chromadb.PersistentClient(path=str(directory))
    collection = client.get_collection(name=collection_name)
    return collection.configuration_json["hnsw"]["space"]


def print_metrics(
    manual_metrics: dict[str, Any],
    langchain_metrics: dict[str, Any],
) -> None:
    print("\nVector Retrieval Regression Summary")
    print("=" * 72)
    print(f"{'Metric':<12}{'Manual':<25}{'LangChain':<25}")
    print("-" * 72)
    for top_k in HIT_K_VALUES:
        metric_name = f"hit_at_{top_k}"
        manual = manual_metrics[metric_name]
        langchain = langchain_metrics[metric_name]
        manual_text = (
            f"{manual['hit_count']}/{manual_metrics['total_cases']} "
            f"({manual['hit_rate']:.2%})"
        )
        langchain_text = (
            f"{langchain['hit_count']}/{langchain_metrics['total_cases']} "
            f"({langchain['hit_rate']:.2%})"
        )
        print(
            f"Hit@{top_k:<7}"
            f"{manual_text:<25}"
            f"{langchain_text:<25}"
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    manual_client = chromadb.PersistentClient(
        path=str(SOURCE_CHROMA_DIR)
    )
    manual_collection = manual_client.get_collection(
        name=SOURCE_COLLECTION_NAME
    )
    vectorstore = initialize_vectorstore()
    embedding_adapter = ExistingEmbeddingAdapter()
    test_cases = load_test_cases()

    manual_space = get_collection_space(
        SOURCE_CHROMA_DIR,
        SOURCE_COLLECTION_NAME,
    )
    langchain_space = get_collection_space(
        LANGCHAIN_CHROMA_DIR,
        LANGCHAIN_COLLECTION_NAME,
    )
    collection_comparison = compare_collections(
        manual_collection,
        vectorstore,
    )

    details = []
    manual_hit_ranks = []
    langchain_hit_ranks = []

    for case in test_cases:
        manual_candidates, manual_embedding = query_manual_vector(
            manual_collection,
            case["question"],
            top_k=TOP_K,
        )
        langchain_candidates = query_langchain_vector(
            vectorstore,
            case["question"],
            top_k=TOP_K,
        )
        langchain_embedding = embedding_adapter.embed_query(
            case["question"]
        )

        manual_hit_rank = find_hit_rank(
            manual_candidates,
            case["expected_source"],
            case["expected_page"],
        )
        langchain_hit_rank = find_hit_rank(
            langchain_candidates,
            case["expected_source"],
            case["expected_page"],
        )
        candidate_comparison = compare_candidates(
            manual_candidates,
            langchain_candidates,
        )
        embedding_comparison = compare_query_embeddings(
            manual_embedding,
            langchain_embedding,
        )

        manual_hit_ranks.append(manual_hit_rank)
        langchain_hit_ranks.append(langchain_hit_rank)
        details.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_source": case["expected_source"],
                "expected_page": case["expected_page"],
                "manual_hit_rank": manual_hit_rank,
                "langchain_hit_rank": langchain_hit_rank,
                "manual_top10": serialize_candidates(manual_candidates),
                "langchain_top10": serialize_candidates(
                    langchain_candidates
                ),
                "query_embedding_comparison": embedding_comparison,
                "retrieval_comparison": candidate_comparison,
            }
        )

        status = (
            "MATCH"
            if (
                manual_hit_rank == langchain_hit_rank
                and embedding_comparison["match"]
                and candidate_comparison["top_k_id_order_match"]
                and candidate_comparison["metadata_match"]
                and candidate_comparison["chunk_content_match"]
                and candidate_comparison["vector_distances_match"]
            )
            else "DIFF"
        )
        print(
            f"{case['id']}: manual_rank={manual_hit_rank}, "
            f"langchain_rank={langchain_hit_rank}, {status}"
        )

    manual_metrics = build_metrics(manual_hit_ranks)
    langchain_metrics = build_metrics(langchain_hit_ranks)
    order_difference_cases = [
        detail["id"]
        for detail in details
        if not detail["retrieval_comparison"]["top_k_id_order_match"]
    ]
    metadata_difference_cases = [
        detail["id"]
        for detail in details
        if not detail["retrieval_comparison"]["metadata_match"]
    ]
    content_difference_cases = [
        detail["id"]
        for detail in details
        if not detail["retrieval_comparison"]["chunk_content_match"]
    ]
    distance_difference_cases = [
        detail["id"]
        for detail in details
        if not detail["retrieval_comparison"]["vector_distances_match"]
    ]
    embedding_difference_cases = [
        detail["id"]
        for detail in details
        if not detail["query_embedding_comparison"]["match"]
    ]
    hit_rank_difference_cases = [
        detail["id"]
        for detail in details
        if detail["manual_hit_rank"] != detail["langchain_hit_rank"]
    ]

    output = {
        "experiment": "Manual Vector Retrieval vs LangChain Vector Retrieval",
        "parameters": {
            "test_case_file": str(TEST_CASE_FILE.relative_to(PROJECT_ROOT)),
            "top_k": TOP_K,
            "hit_k_values": list(HIT_K_VALUES),
            "manual_collection": SOURCE_COLLECTION_NAME,
            "manual_persist_directory": str(
                SOURCE_CHROMA_DIR.relative_to(PROJECT_ROOT)
            ),
            "langchain_collection": LANGCHAIN_COLLECTION_NAME,
            "langchain_persist_directory": str(
                LANGCHAIN_CHROMA_DIR.relative_to(PROJECT_ROOT)
            ),
            "distance_space": DISTANCE_SPACE,
            "manual_distance_space": manual_space,
            "langchain_distance_space": langchain_space,
            "embedding_adapter": "ExistingEmbeddingAdapter",
            "hybrid_enabled": False,
            "reranker_enabled": False,
            "query_rewrite_enabled": False,
            "query_expansion_enabled": False,
        },
        "preflight_consistency": {
            **collection_comparison,
            "distance_space_match": (
                manual_space == langchain_space == DISTANCE_SPACE
            ),
        },
        "summary": {
            "manual": manual_metrics,
            "langchain": langchain_metrics,
            "hit_rank_difference_cases": hit_rank_difference_cases,
            "top_k_order_difference_cases": order_difference_cases,
            "metadata_difference_cases": metadata_difference_cases,
            "chunk_content_difference_cases": content_difference_cases,
            "query_embedding_difference_cases": (
                embedding_difference_cases
            ),
            "vector_distance_difference_cases": (
                distance_difference_cases
            ),
        },
        "details": details,
    }

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print_metrics(manual_metrics, langchain_metrics)
    print("\nConsistency Checks")
    print("=" * 72)
    print(f"Collection data: {collection_comparison}")
    print(
        "Distance space: "
        f"manual={manual_space}, "
        f"langchain={langchain_space}, "
        f"match={manual_space == langchain_space == DISTANCE_SPACE}"
    )
    print(f"Hit-rank differences: {hit_rank_difference_cases}")
    print(f"Top-10 order differences: {order_difference_cases}")
    print(f"Metadata differences: {metadata_difference_cases}")
    print(f"Chunk-content differences: {content_difference_cases}")
    print(f"Query-embedding differences: {embedding_difference_cases}")
    print(f"Vector-distance differences: {distance_difference_cases}")
    print(
        "Result saved: "
        f"{RESULT_FILE.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
