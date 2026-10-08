import json
import sys

from langchain_rag.lc_rag import ask_langchain_rag
from langchain_rag.lc_vectorstore import PROJECT_ROOT


TEST_CASE_FILE = PROJECT_ROOT / "eval" / "test_case.json"
TEST_CASE_ID = "Q002"
TOP_K = 3


def load_test_case(case_id: str) -> dict:
    with TEST_CASE_FILE.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    for case in test_cases:
        if case["id"] == case_id:
            return case
    raise ValueError(f"Test case not found: {case_id}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    case = load_test_case(TEST_CASE_ID)
    result = ask_langchain_rag(case["question"], top_k=TOP_K)

    print("Question")
    print("=" * 72)
    print(result["question"])

    print("\nRetrieved Documents")
    print("=" * 72)
    for rank, document in enumerate(result["documents"], start=1):
        metadata = document["metadata"]
        print(f"Rank: {rank}")
        print(f"Source: {metadata.get('source')}")
        print(f"Page: {metadata.get('page')}")
        print(f"Chunk ID: {metadata.get('chunk_id')}")
        print("Document Content:")
        print(document["page_content"])
        print("-" * 72)

    print("\nContext")
    print("=" * 72)
    print(result["context"])

    print("\nAnswer")
    print("=" * 72)
    print(result["answer"])

    print("\nCitations (from retrieved Metadata)")
    print("=" * 72)
    print(json.dumps(result["sources"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
