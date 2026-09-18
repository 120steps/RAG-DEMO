import json
import sys
from pathlib import Path

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedding import embed_text


TEST_CASE_FILE = PROJECT_ROOT / "eval" / "test_case.json"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
RESULT_FILE = PROJECT_ROOT / "eval" / "results" / "miss_cases.json"
COLLECTION_NAME = "company_knowledge"
TOP_K = 10


def load_answerable_test_cases():
    with TEST_CASE_FILE.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    return [
        case
        for case in test_cases
        if case.get("should_answer", True)
    ]


def find_miss_cases(collection, test_cases):
    miss_cases = []

    for case in test_cases:
        expected_source = case.get("expected_source")
        expected_page = case.get("expected_page")
        question_embedding = embed_text(case["question"])
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        hit = any(
            metadata.get("source") == expected_source
            and metadata.get("page") == expected_page
            for metadata in metadatas
        )

        if hit:
            continue

        retrieved = [
            {
                "rank": rank,
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "chunk_id": metadata.get("chunk_id"),
                "distance": float(distance),
                "document": document,
            }
            for rank, (document, metadata, distance) in enumerate(
                zip(documents, metadatas, distances),
                start=1,
            )
        ]
        miss_cases.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_source": expected_source,
                "expected_page": expected_page,
                "retrieved": retrieved,
            }
        )

    return miss_cases


def print_miss_case(miss_case):
    print()
    print("=" * 80)
    print(f"id: {miss_case['id']}")
    print(f"question: {miss_case['question']}")
    print(f"expected_source: {miss_case['expected_source']}")
    print(f"expected_page: {miss_case['expected_page']}")

    for item in miss_case["retrieved"]:
        print()
        print(f"Top {item['rank']}")
        print(f"source: {item['source']}")
        print(f"page: {item['page']}")
        print(f"chunk_id: {item['chunk_id']}")
        print(f"distance: {item['distance']:.4f}")
        print(f"document: {item['document']}")


def save_results(test_cases, miss_cases):
    miss_case_ids = [case["id"] for case in miss_cases]
    result = {
        "summary": {
            "total_answerable_cases": len(test_cases),
            "top_k": TOP_K,
            "miss_count": len(miss_cases),
            "miss_case_ids": miss_case_ids,
        },
        "miss_cases": miss_cases,
    }

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_FILE.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return miss_case_ids


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    test_cases = load_answerable_test_cases()
    miss_cases = find_miss_cases(collection, test_cases)

    for miss_case in miss_cases:
        print_miss_case(miss_case)

    miss_case_ids = save_results(test_cases, miss_cases)

    print()
    print("Retrieval Miss Summary")
    print("=" * 80)
    print(f"Total Misses: {len(miss_cases)}")
    print(f"Miss Case IDs: {miss_case_ids}")
    print(f"Result saved: {RESULT_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
