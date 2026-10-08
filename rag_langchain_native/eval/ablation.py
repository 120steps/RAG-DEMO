"""V3 retrieval component ablation benchmark."""

from __future__ import annotations

import argparse
from dataclasses import replace

from ..config import DEFAULT_SETTINGS
from .common import RESULTS_DIR, write_json
from .retrieval_eval import run_retrieval_evaluation


MODES = {
    "vector_only": dict(
        vector_enabled=True, bm25_enabled=False, reranker_enabled=False,
        rewrite_enabled=False, expansion_enabled=False,
    ),
    "hybrid": dict(
        vector_enabled=True, bm25_enabled=True, reranker_enabled=False,
        rewrite_enabled=False, expansion_enabled=False,
    ),
    "hybrid_reranker": dict(
        vector_enabled=True, bm25_enabled=True, reranker_enabled=True,
        rewrite_enabled=False, expansion_enabled=False,
    ),
    "rewrite_hybrid_reranker": dict(
        vector_enabled=True, bm25_enabled=True, reranker_enabled=True,
        rewrite_enabled=True, expansion_enabled=False,
    ),
    "rewrite_expansion_hybrid_reranker": dict(
        vector_enabled=True, bm25_enabled=True, reranker_enabled=True,
        rewrite_enabled=True, expansion_enabled=True,
    ),
}


def run_ablation(mode_names: list[str] | None = None) -> dict:
    results = {}
    selected = mode_names or list(MODES)
    for mode in selected:
        overrides = MODES[mode]
        print(f"Running {mode}...", flush=True)
        settings = replace(DEFAULT_SETTINGS, final_k=10, **overrides)
        path = RESULTS_DIR / f"ablation_{mode}.json"
        output = run_retrieval_evaluation(
            settings=settings,
            result_file=path,
        )
        results[mode] = output["summary"]
    skipped = [mode for mode in MODES if mode not in selected]
    summary = {
        "modes": results,
        "skipped_modes": skipped,
        "skip_reason": (
            "Gemini network execution was not authorized"
            if skipped else None
        ),
    }
    write_json(RESULTS_DIR / "ablation_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Run modes that do not require Gemini query processing",
    )
    args = parser.parse_args()
    local_modes = ["vector_only", "hybrid", "hybrid_reranker"]
    result = run_ablation(local_modes if args.local_only else None)
    for name, summary in result["modes"].items():
        values = " ".join(
            f"Hit@{k}={summary[f'hit_at_{k}']['hit_rate']:.2%}"
            for k in (1, 3, 5, 10)
        )
        print(f"{name}: {values}; MRR={summary['mrr']:.4f}")

