import json
import math
import sys
from pathlib import Path

import chromadb

from embedding import embed_text
from langchain_rag.lc_embedding import ExistingEmbeddingAdapter
from langchain_rag.lc_retriever import retrieve
from langchain_rag.lc_vectorstore import (
    DISTANCE_SPACE,
    LANGCHAIN_CHROMA_DIR,
    LANGCHAIN_COLLECTION_NAME,
    PROJECT_ROOT,
    SOURCE_CHROMA_DIR,
    SOURCE_COLLECTION_NAME,
    initialize_vectorstore,
    similarity_search,
    similarity_search_with_distance,
)


TEST_CASE_FILE = PROJECT_ROOT / "eval" / "test_case.json"
TEST_CASE_ID = "Q002"
TOP_K = 5
DISTANCE_TOLERANCE = 1e-6
EMBEDDING_TOLERANCE = 1e-7


def load_test_case(case_id):
    with TEST_CASE_FILE.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    for case in test_cases:
        if case["id"] == case_id:
            return case
    raise ValueError(f"Test case not found: {case_id}")


def query_original_vector_store(query, top_k):
    client = chromadb.PersistentClient(path=str(SOURCE_CHROMA_DIR))
    collection = client.get_collection(name=SOURCE_COLLECTION_NAME)
    result = collection.query(
        query_embeddings=[embed_text(query)],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
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


def get_collection_space(directory: Path, collection_name: str):
    client = chromadb.PersistentClient(path=str(directory))
    collection = client.get_collection(name=collection_name)
    return collection.configuration_json["hnsw"]["space"]


def compare_embeddings(query):
    original = embed_text(query)
    adapted = ExistingEmbeddingAdapter().embed_query(query)
    differences = [
        abs(left - right)
        for left, right in zip(original, adapted)
    ]
    max_difference = max(differences, default=0.0)
    same_dimension = len(original) == len(adapted)
    passed = same_dimension and max_difference <= EMBEDDING_TOLERANCE
    return {
        "original_dimension": len(original),
        "langchain_dimension": len(adapted),
        "max_absolute_difference": max_difference,
        "passed": passed,
    }


def print_langchain_results(results):
    print("LangChain Vector Search Top-5")
    print("=" * 72)
    for rank, (document, distance) in enumerate(results, start=1):
        metadata = document.metadata
        print(f"Rank: {rank}")
        print(f"Source: {metadata.get('source')}")
        print(f"Page: {metadata.get('page')}")
        print(f"Chunk ID: {metadata.get('chunk_id')}")
        print(f"Stable ID: {document.id}")
        print(f"Distance: {distance:.12f}")
        print("Document Content:")
        print(document.page_content)
        print("-" * 72)


def compare_results(original_results, langchain_results):
    langchain_records = [
        {
            "id": document.id,
            "document": document.page_content,
            "metadata": document.metadata,
            "distance": float(distance),
        }
        for document, distance in langchain_results
    ]

    rank_ids_match = [
        result["id"] for result in original_results
    ] == [result["id"] for result in langchain_records]
    documents_match = all(
        original["document"] == adapted["document"]
        for original, adapted in zip(
            original_results,
            langchain_records,
        )
    )
    metadata_match = all(
        original["metadata"] == adapted["metadata"]
        for original, adapted in zip(
            original_results,
            langchain_records,
        )
    )
    distance_differences = [
        abs(original["distance"] - adapted["distance"])
        for original, adapted in zip(
            original_results,
            langchain_records,
        )
    ]
    max_distance_difference = max(
        distance_differences,
        default=math.inf,
    )
    distances_match = max_distance_difference <= DISTANCE_TOLERANCE
    top_k_match = (
        len(original_results) == TOP_K
        and len(langchain_records) == TOP_K
    )

    return {
        "top_k_match": top_k_match,
        "rank_ids_match": rank_ids_match,
        "documents_match": documents_match,
        "metadata_match": metadata_match,
        "max_distance_difference": max_distance_difference,
        "distances_match": distances_match,
        "passed": all(
            (
                top_k_match,
                rank_ids_match,
                documents_match,
                metadata_match,
                distances_match,
            )
        ),
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    case = load_test_case(TEST_CASE_ID)
    question = case["question"]

    vectorstore = initialize_vectorstore()
    source_count = chromadb.PersistentClient(
        path=str(SOURCE_CHROMA_DIR)
    ).get_collection(name=SOURCE_COLLECTION_NAME).count()
    langchain_count = len(vectorstore.get(include=[])["ids"])

    langchain_results = similarity_search_with_distance(
        question,
        top_k=TOP_K,
    )
    original_results = query_original_vector_store(
        question,
        top_k=TOP_K,
    )
    similarity_documents = similarity_search(
        question,
        top_k=TOP_K,
    )
    retriever_documents = retrieve(question, top_k=TOP_K)

    print(f"Test Case: {case['id']}")
    print(f"Question: {question}")
    print(f"Top-K: {TOP_K}")
    print(f"Source collection count: {source_count}")
    print(f"LangChain collection count: {langchain_count}")
    print()
    print_langchain_results(langchain_results)

    embedding_comparison = compare_embeddings(question)
    retrieval_comparison = compare_results(
        original_results,
        langchain_results,
    )
    original_space = get_collection_space(
        SOURCE_CHROMA_DIR,
        SOURCE_COLLECTION_NAME,
    )
    langchain_space = get_collection_space(
        LANGCHAIN_CHROMA_DIR,
        LANGCHAIN_COLLECTION_NAME,
    )
    distance_space_match = (
        original_space == langchain_space == DISTANCE_SPACE
    )
    similarity_api_match = [
        document.id for document in similarity_documents
    ] == [document.id for document, _ in langchain_results]
    retriever_match = [
        document.id for document in retriever_documents
    ] == [document.id for document, _ in langchain_results]
    count_match = source_count == langchain_count

    print("Consistency Check")
    print("=" * 72)
    print(f"Embedding: {embedding_comparison}")
    print(f"Chunk count match: {count_match}")
    print(
        "Distance space: "
        f"original={original_space}, "
        f"langchain={langchain_space}, "
        f"match={distance_space_match}"
    )
    print(f"Vector result comparison: {retrieval_comparison}")
    print(f"similarity_search order match: {similarity_api_match}")
    print(f"Retriever order match: {retriever_match}")

    passed = all(
        (
            embedding_comparison["passed"],
            count_match,
            distance_space_match,
            retrieval_comparison["passed"],
            similarity_api_match,
            retriever_match,
        )
    )
    print(f"Final consistency test: {'PASS' if passed else 'FAIL'}")

    if not passed:
        raise AssertionError(
            "LangChain Vector Search does not match the original pure "
            "Vector Search."
        )


if __name__ == "__main__":
    main()
