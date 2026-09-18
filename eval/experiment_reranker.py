import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieve_eval import evaluate_retrieval


RESULT_DIR = PROJECT_ROOT / "eval" / "results"


def case_changes(without_reranker, with_reranker, metric):
    off_by_id = {
        case["id"]: case
        for case in without_reranker["details"]
    }
    on_by_id = {
        case["id"]: case
        for case in with_reranker["details"]
    }
    improved = []
    worse = []

    for case_id, off_case in off_by_id.items():
        off_hit = off_case[metric]
        on_hit = on_by_id[case_id][metric]
        if not off_hit and on_hit:
            improved.append(case_id)
        elif off_hit and not on_hit:
            worse.append(case_id)

    return improved, worse


def main():
    without_reranker = evaluate_retrieval(
        use_reranker=False,
        result_file=RESULT_DIR / "reranker_off.json",
        verbose=False,
    )
    with_reranker = evaluate_retrieval(
        use_reranker=True,
        result_file=RESULT_DIR / "reranker_on.json",
        verbose=False,
    )

    off_summary = without_reranker["summary"]
    on_summary = with_reranker["summary"]
    top1_improved, top1_worse = case_changes(
        without_reranker,
        with_reranker,
        "top1_hit",
    )
    hit3_improved, hit3_worse = case_changes(
        without_reranker,
        with_reranker,
        "hit_at_3",
    )

    print("Reranker Experiment Summary")
    print("=" * 72)
    print(
        f"{'Mode':<20} "
        f"{'Top1 Hits':>10} "
        f"{'Top1 Rate':>11} "
        f"{'Hit@3':>10} "
        f"{'Hit@3 Rate':>12}"
    )
    print("-" * 72)
    print(
        f"{'Without Reranker':<20} "
        f"{off_summary['top1_hit_count']:>10} "
        f"{off_summary['top1_hit_rate']:>10.2%} "
        f"{off_summary['hit_at_3_count']:>10} "
        f"{off_summary['hit_at_3_rate']:>11.2%}"
    )
    print(
        f"{'With Reranker':<20} "
        f"{on_summary['top1_hit_count']:>10} "
        f"{on_summary['top1_hit_rate']:>10.2%} "
        f"{on_summary['hit_at_3_count']:>10} "
        f"{on_summary['hit_at_3_rate']:>11.2%}"
    )

    top1_delta = (
        on_summary["top1_hit_rate"]
        - off_summary["top1_hit_rate"]
    )
    hit3_delta = (
        on_summary["hit_at_3_rate"]
        - off_summary["hit_at_3_rate"]
    )
    print()
    print(f"Top1 change: {top1_delta:+.2%}")
    print(f"Hit@3 change: {hit3_delta:+.2%}")
    print(f"Top1 improved cases: {top1_improved}")
    print(f"Top1 worse cases: {top1_worse}")
    print(f"Hit@3 improved cases: {hit3_improved}")
    print(f"Hit@3 worse cases: {hit3_worse}")


if __name__ == "__main__":
    main()
