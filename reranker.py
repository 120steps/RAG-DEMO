from functools import lru_cache
from pathlib import Path

from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MODEL_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "huggingface"


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(
        RERANKER_MODEL,
        max_length=512,
        cache_folder=str(MODEL_CACHE_DIR),
    )


def bind_candidates(documents, metadatas, distances):
    if not (
        len(documents)
        == len(metadatas)
        == len(distances)
    ):
        raise ValueError(
            "documents, metadatas, and distances must have equal lengths"
        )

    return [
        {
            "document": document,
            "metadata": metadata,
            "original_distance": float(distance),
            "rerank_score": None,
        }
        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        )
    ]


def rerank_candidates(question, candidates, top_n=3):
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    if not candidates:
        return []

    pairs = [
        (question, candidate["document"])
        for candidate in candidates
    ]
    scores = get_reranker().predict(
        pairs,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    reranked = [
        {
            **candidate,
            "rerank_score": float(score),
        }
        for candidate, score in zip(candidates, scores)
    ]
    reranked.sort(
        key=lambda candidate: candidate["rerank_score"],
        reverse=True,
    )
    return reranked[:top_n]
