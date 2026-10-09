from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

from ..config import PACKAGE_DIR, PROJECT_ROOT


TEST_CASE_FILE = PROJECT_ROOT / "eval" / "test_case.json"
RESULTS_DIR = PACKAGE_DIR / "eval" / "results"


def load_test_cases(*, answerable_only: bool = False) -> list[dict]:
    cases = json.loads(TEST_CASE_FILE.read_text(encoding="utf-8"))
    if answerable_only:
        return [case for case in cases if case.get("should_answer") is True]
    return cases


def expected_rank(case: dict, documents: list[dict]) -> int | None:
    for index, document in enumerate(documents, start=1):
        if (
            document.get("source") == case.get("expected_source")
            and document.get("page") == case.get("expected_page")
        ):
            return index
    return None


def percentile95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                path.write_text(
                    temporary.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                temporary.unlink(missing_ok=True)
                return
            time.sleep(0.2 * (attempt + 1))


def retrieval_summary(details: list[dict]) -> dict:
    total = len(details)
    latencies = [float(item["latency_ms"]) for item in details]
    summary = {
        "total_cases": total,
        "mrr": sum(
            1.0 / item["hit_rank"] if item["hit_rank"] else 0.0
            for item in details
        ) / total if total else 0.0,
        "average_retrieval_latency_ms": statistics.fmean(latencies)
        if latencies else 0.0,
        "p95_retrieval_latency_ms": percentile95(latencies),
        "miss_cases": [item["id"] for item in details if item["hit_rank"] is None],
    }
    for k in (1, 3, 5, 10):
        hits = sum(
            item["hit_rank"] is not None and item["hit_rank"] <= k
            for item in details
        )
        summary[f"hit_at_{k}"] = {
            "hit_count": hits,
            "hit_rate": hits / total if total else 0.0,
        }
    return summary

