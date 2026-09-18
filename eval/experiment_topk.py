import json
import sys
from pathlib import Path

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHUNK_OVERLAP, CHUNK_SIZE
from document_loader import load_pdf
from embedding import embed_texts


EXPECTED_CHUNK_SIZE = 200
EXPECTED_CHUNK_OVERLAP = 30
TOP_K_VALUES = (1, 3, 5, 10)

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
PDF_DIR = PROJECT_ROOT / "data" / "pdf"
TEST_CASE_FILE = PROJECT_ROOT / "eval" / "test_case.json"
RESULT_DIR = PROJECT_ROOT / "eval" / "results"
COLLECTION_NAME = "company_knowledge"


def rebuild_collection(client):
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    chunks = []
    metadatas = []
    ids = []

    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        documents = load_pdf(
            str(pdf_path),
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        for document in documents:
            ids.append(
                f"{document['source']}"
                f"_page_{document['page']}"
                f"_chunk_{document['chunk_id']}"
            )
            chunks.append(document["text"])
            metadatas.append(
                {
                    "source": document["source"],
                    "page": document["page"],
                    "chunk_id": document["chunk_id"],
                }
            )

    embeddings = embed_texts(chunks)
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return collection, len(chunks)


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


def evaluate(collection, test_cases, question_embeddings, top_k):
    hit_count = 0
    details = []
    miss_cases = []

    for case, question_embedding in zip(test_cases, question_embeddings):
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
        )
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        expected_source = case["expected_source"]
        expected_page = case["expected_page"]
        hit_rank = None

        retrieved = []
        for rank, (metadata, distance) in enumerate(
            zip(metadatas, distances),
            start=1,
        ):
            retrieved.append(
                {
                    "rank": rank,
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "distance": float(distance),
                }
            )

            if (
                hit_rank is None
                and metadata.get("source") == expected_source
                and metadata.get("page") == expected_page
            ):
                hit_rank = rank

        hit = hit_rank is not None
        if hit:
            hit_count += 1
        else:
            miss_cases.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "expected_source": expected_source,
                    "expected_page": expected_page,
                }
            )

        details.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_source": expected_source,
                "expected_page": expected_page,
                "hit": hit,
                "hit_rank": hit_rank,
                "retrieved": retrieved,
            }
        )

    total_cases = len(test_cases)
    summary = {
        "hit_count": hit_count,
        "total_cases": total_cases,
        "miss_count": total_cases - hit_count,
        "hit_rate": hit_count / total_cases if total_cases else 0,
    }
    return summary, details, miss_cases


def save_result(top_k, chunk_count, summary, details):
    result = {
        "parameters": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": top_k,
        },
        "collection": {
            "name": COLLECTION_NAME,
            "chunk_count": chunk_count,
        },
        "summary": summary,
        "details": details,
    }
    output_path = RESULT_DIR / f"topk_{top_k}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return output_path


def print_miss_cases(top_k, miss_cases):
    print()
    print(f"Miss Cases for Hit@{top_k}")
    print("-" * 80)

    if not miss_cases:
        print("None")
        return

    for case in miss_cases:
        print(f"id: {case['id']}")
        print(f"question: {case['question']}")
        print(f"expected_source: {case['expected_source']}")
        print(f"expected_page: {case['expected_page']}")
        print()


def print_summary(experiment_results):
    print()
    print("Top-K Retrieval Experiment Summary")
    print("=" * 58)
    print(
        f"{'Metric':>10}  "
        f"{'hit_count':>10}  "
        f"{'total_cases':>11}  "
        f"{'hit_rate':>10}"
    )
    print("-" * 58)

    for top_k, summary in experiment_results:
        print(
            f"{f'Hit@{top_k}':>10}  "
            f"{summary['hit_count']:>10}  "
            f"{summary['total_cases']:>11}  "
            f"{summary['hit_rate']:>9.2%}"
        )


def main():
    if (
        CHUNK_SIZE != EXPECTED_CHUNK_SIZE
        or CHUNK_OVERLAP != EXPECTED_CHUNK_OVERLAP
    ):
        raise ValueError(
            "Top-K experiment requires "
            f"CHUNK_SIZE={EXPECTED_CHUNK_SIZE} and "
            f"CHUNK_OVERLAP={EXPECTED_CHUNK_OVERLAP}; "
            f"current values are {CHUNK_SIZE}/{CHUNK_OVERLAP}."
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    test_cases = load_test_cases()
    question_embeddings = embed_texts(
        [case["question"] for case in test_cases]
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    print(
        f"Rebuilding Chroma once with chunk_size={CHUNK_SIZE}, "
        f"chunk_overlap={CHUNK_OVERLAP}"
    )
    collection, chunk_count = rebuild_collection(client)
    print(f"Chunks rebuilt: {chunk_count}")

    experiment_results = []
    for top_k in TOP_K_VALUES:
        summary, details, miss_cases = evaluate(
            collection,
            test_cases,
            question_embeddings,
            top_k,
        )
        output_path = save_result(
            top_k,
            chunk_count,
            summary,
            details,
        )
        experiment_results.append((top_k, summary))
        print(f"Hit@{top_k} result saved: {output_path.relative_to(PROJECT_ROOT)}")
        print_miss_cases(top_k, miss_cases)

    print_summary(experiment_results)


if __name__ == "__main__":
    main()
