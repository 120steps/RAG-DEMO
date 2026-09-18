import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieve_eval import evaluate_retrieval


RESULT_DIR = PROJECT_ROOT / "eval" / "results"
EXPERIMENTS = (
    {
        "name": "Vector Only",
        "use_hybrid": False,
        "use_reranker": False,
        "filename": "hybrid_vector_only.json",
    },
    {
        "name": "Vector + Reranker",
        "use_hybrid": False,
        "use_reranker": True,
        "filename": "hybrid_vector_reranker.json",
    },
    {
        "name": "Hybrid + Reranker",
        "use_hybrid": True,
        "use_reranker": True,
        "filename": "hybrid_vector_bm25_reranker.json",
    },
)


def main():
    experiment_results = []

    for experiment in EXPERIMENTS:
        print(f"Running: {experiment['name']}")
        result = evaluate_retrieval(
            use_reranker=experiment["use_reranker"],
            use_hybrid=experiment["use_hybrid"],
            final_top_k=5,
            result_file=RESULT_DIR / experiment["filename"],
            verbose=False,
        )
        experiment_results.append(
            (experiment["name"], result["summary"])
        )

    print()
    print("Hybrid Search Experiment Summary")
    print("=" * 89)
    print(
        f"{'Mode':<22} "
        f"{'Top1 Hits':>10} "
        f"{'Top1 Rate':>11} "
        f"{'Hit@3':>8} "
        f"{'Hit@3 Rate':>11} "
        f"{'Hit@5':>8} "
        f"{'Hit@5 Rate':>11}"
    )
    print("-" * 89)

    for name, summary in experiment_results:
        print(
            f"{name:<22} "
            f"{summary['top1_hit_count']:>10} "
            f"{summary['top1_hit_rate']:>10.2%} "
            f"{summary['hit_at_3_count']:>8} "
            f"{summary['hit_at_3_rate']:>10.2%} "
            f"{summary['hit_at_5_count']:>8} "
            f"{summary['hit_at_5_rate']:>10.2%}"
        )


if __name__ == "__main__":
    main()
