from config import (
    ANSWERABILITY_MAX_VECTOR_DISTANCE,
    ANSWERABILITY_MIN_BM25_SCORE,
    ANSWERABILITY_MIN_RERANK_SCORE,
)


def assess_answerability(candidates):
    thresholds = {
        "min_rerank_score": ANSWERABILITY_MIN_RERANK_SCORE,
        "min_bm25_score": ANSWERABILITY_MIN_BM25_SCORE,
        "max_vector_distance": ANSWERABILITY_MAX_VECTOR_DISTANCE,
    }

    if not candidates:
        return {
            "should_answer": False,
            "reason": "no_candidates",
            "passed_signals": [],
            "top_candidate_scores": {
                "rerank_score": None,
                "bm25_score": None,
                "vector_distance": None,
            },
            "thresholds": thresholds,
        }

    top_candidate = candidates[0]
    rerank_score = top_candidate.get("rerank_score")
    bm25_score = top_candidate.get("bm25_score")
    vector_distance = top_candidate.get("original_distance")
    passed_signals = []

    if (
        rerank_score is not None
        and rerank_score >= ANSWERABILITY_MIN_RERANK_SCORE
    ):
        passed_signals.append("rerank_score")

    if (
        bm25_score is not None
        and bm25_score >= ANSWERABILITY_MIN_BM25_SCORE
    ):
        passed_signals.append("bm25_score")

    if (
        vector_distance is not None
        and vector_distance <= ANSWERABILITY_MAX_VECTOR_DISTANCE
    ):
        passed_signals.append("vector_distance")

    should_answer = bool(passed_signals)
    return {
        "should_answer": should_answer,
        "reason": (
            "sufficient_retrieval_evidence"
            if should_answer
            else "insufficient_retrieval_evidence"
        ),
        "passed_signals": passed_signals,
        "top_candidate_scores": {
            "rerank_score": rerank_score,
            "bm25_score": bm25_score,
            "vector_distance": vector_distance,
        },
        "thresholds": thresholds,
    }
