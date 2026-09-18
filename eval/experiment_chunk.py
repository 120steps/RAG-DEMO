import json
import sys
from pathlib import Path

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TOP_K
from document_loader import load_pdf
from embedding import embed_text, embed_texts


PDF_DIR = PROJECT_ROOT / "data" / "pdf"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
TEST_CASE_FILE = PROJECT_ROOT / "eval" / "test_case.json"
RESULT_DIR = PROJECT_ROOT / "eval" / "results"
COLLECTION_NAME = "company_knowledge"

EXPERIMENTS = (
    (100, 20, "chunk_100_20.json"),
    (300, 60, "chunk_300_60.json"),
    (500, 100, "chunk_500_100.json"),
)


def rebuild_collection(client, chunk_size, chunk_overlap):
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
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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


def evaluate(collection, test_cases):
    hit_count = 0
    details = []

    for case in test_cases:
        question_embedding = embed_text(case["question"])
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=TOP_K,
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
        "total_cases": total_cases,
        "hit_cases": hit_count,
        "miss_cases": total_cases - hit_count,
        "hit_rate": hit_count / total_cases if total_cases else 0,
    }
    return summary, details


def save_result(
    filename,
    chunk_size,
    chunk_overlap,
    chunk_count,
    summary,
    details,
):
    result = {
        "parameters": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": TOP_K,
        },
        "collection": {
            "name": COLLECTION_NAME,
            "chunk_count": chunk_count,
        },
        "summary": summary,
        "details": details,
    }
    output_path = RESULT_DIR / filename
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return output_path


def print_summary(experiment_results):
    print()
    print("Retrieval Chunk Experiment Summary")
    print("=" * 69)
    print(
        f"{'chunk_size':>10}  "
        f"{'overlap':>8}  "
        f"{'hit_count':>10}  "
        f"{'total_cases':>11}  "
        f"{'hit_rate':>10}"
    )
    print("-" * 69)

    for result in experiment_results:
        print(
            f"{result['chunk_size']:>10}  "
            f"{result['chunk_overlap']:>8}  "
            f"{result['hit_count']:>10}  "
            f"{result['total_cases']:>11}  "
            f"{result['hit_rate']:>9.2%}"
        )


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    test_cases = load_test_cases()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    experiment_results = []

    for chunk_size, chunk_overlap, filename in EXPERIMENTS:
        print()
        print(
            f"Running chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )
        collection, chunk_count = rebuild_collection(
            client,
            chunk_size,
            chunk_overlap,
        )
        summary, details = evaluate(collection, test_cases)
        output_path = save_result(
            filename,
            chunk_size,
            chunk_overlap,
            chunk_count,
            summary,
            details,
        )
        experiment_results.append(
            {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "hit_count": summary["hit_cases"],
                "total_cases": summary["total_cases"],
                "hit_rate": summary["hit_rate"],
            }
        )
        print(f"Chunks rebuilt: {chunk_count}")
        print(f"Result saved: {output_path.relative_to(PROJECT_ROOT)}")

    print_summary(experiment_results)


if __name__ == "__main__":
    main()
