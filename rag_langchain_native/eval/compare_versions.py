"""Read-only comparison of V1/V2 historical benchmark and current V3 results."""

from __future__ import annotations

import json

from ..config import DEFAULT_SETTINGS, PROJECT_ROOT
from .common import RESULTS_DIR, write_json


OLD_RESULT = PROJECT_ROOT / "eval" / "results" / "langchain_full_benchmark.json"
V3_RETRIEVAL = RESULTS_DIR / "retrieval_benchmark.json"
V3_ANSWER = RESULTS_DIR / "answer_benchmark.json"
RESULT_FILE = RESULTS_DIR / "version_comparison.json"


def _read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def compare_versions() -> dict:
    old = _read(OLD_RESULT)
    v3_retrieval = _read(V3_RETRIEVAL)
    v3_answer = _read(V3_ANSWER)
    versions = {}
    if old:
        parameters = old.get("parameters", {})
        retrieval = old.get("retrieval_comparison", {})
        answers = old.get("answer_evaluation", {})
        versions["v1_manual"] = {
            "source": str(OLD_RESULT.relative_to(PROJECT_ROOT)),
            "historical": True,
            "configuration": parameters,
            "retrieval": retrieval.get("manual_summary"),
            "answer": answers.get("manual_summary"),
            "failed_cases": [
                item["id"]
                for item in answers.get("manual_details", [])
                if item.get("answer_correct_deterministic") is False
                or item.get("citation_correct") is False
                or item.get("refusal_correct") is False
            ],
        }
        versions["v2_langchain_wrapper"] = {
            "source": str(OLD_RESULT.relative_to(PROJECT_ROOT)),
            "historical": True,
            "configuration": parameters,
            "retrieval": retrieval.get("langchain_summary"),
            "answer": answers.get("langchain_summary"),
            "failed_cases": [
                item["id"]
                for item in answers.get("langchain_details", [])
                if item.get("answer_correct_deterministic") is False
                or item.get("citation_correct") is False
                or item.get("refusal_correct") is False
            ],
        }
    v3_answer_valid = bool(
        v3_answer and v3_answer.get("valid_for_final_pipeline") is True
    )
    versions["v3_langchain_native"] = {
        "source": "rag_langchain_native/eval/results",
        "historical": False,
        "configuration": {
            "embedding_model": DEFAULT_SETTINGS.embedding_model,
            "embedding_prefixes": [DEFAULT_SETTINGS.query_prefix, DEFAULT_SETTINGS.document_prefix],
            "normalize_embeddings": DEFAULT_SETTINGS.normalize_embeddings,
            "chunk_size": DEFAULT_SETTINGS.chunk_size,
            "chunk_overlap": DEFAULT_SETTINGS.chunk_overlap,
            "collection": DEFAULT_SETTINGS.collection_name,
        },
        "retrieval": v3_retrieval.get("summary") if v3_retrieval else None,
        "answer": v3_answer.get("summary") if v3_answer_valid else None,
        "answer_result_status": (
            "valid"
            if v3_answer_valid
            else "not comparable: final structured pipeline requires a full --force rerun"
        ),
        "failed_cases": (
            v3_answer.get("summary", {}).get("failed_case_ids", [])
            if v3_answer_valid else []
        ),
    }
    output = {
        "strictly_fair_comparison": False,
        "fairness_notes": [
            "V1/V2 values are historical results, not rerun by this script.",
            "V3 uses E5 query/passage prefixes and normalized vectors; V1/V2 historical configuration did not.",
            "V3 uses LangChain RecursiveCharacterTextSplitter, so chunk boundaries may differ even when sizes match.",
            "Latency from different runs and environments is descriptive, not a controlled performance claim.",
        ],
        "versions": versions,
    }
    write_json(RESULT_FILE, output)
    return output


if __name__ == "__main__":
    print(json.dumps(compare_versions(), ensure_ascii=False, indent=2))

