"""汇总已实际生成的 Phase 8/Phase 9 评估文件，不重新运行模型。"""

from __future__ import annotations

import json

from .common import RESULTS_DIR, write_json


RESULT_FILE = RESULTS_DIR / "phase9_summary.json"


def _read(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def run() -> dict:
    baseline = _read("retrieval_benchmark.json")
    retrieval = _read("phase9_retrieval_benchmark.json")
    answer = _read("phase9_answer_benchmark.json")
    judge = _read("phase9_judge_benchmark.json")
    security = _read("phase9_security_evaluation.json")
    output = {
        "phase8_retrieval_baseline": baseline["summary"],
        "phase9_retrieval": retrieval["summary"],
        "phase9_answer": answer["summary"],
        "phase9_judge": judge["summary"],
        "phase9_security": security["summary"],
        "comparability": {
            "retrieval_accuracy": "Comparable model/chunk/retrieval configuration and same ground truth.",
            "latency": "Observed runs were not controlled for cold start or machine load; do not treat as a strict performance claim.",
            "security": "Hit@K is not a security metric; use the separate security evaluation and pytest suite.",
        },
    }
    write_json(RESULT_FILE, output)
    return output


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

