from bm25_search import BM25Index, merge_candidates
from config import TOP_K
from embedding import embed_text
from reranker import bind_candidates, rerank_candidates


VECTOR_RETRIEVAL_K = 10
BM25_RETRIEVAL_K = 10


def retrieve_candidates(
    question,
    collection,
    use_reranker=False,
    use_hybrid=False,
    bm25_index=None,
    final_top_k=TOP_K,
):
    question_embedding = embed_text(question)
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=VECTOR_RETRIEVAL_K,
        include=["documents", "metadatas", "distances"],
    )
    candidates = bind_candidates(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )

    if use_hybrid:
        if bm25_index is None:
            bm25_index = BM25Index.from_chroma(collection)

        bm25_candidates = bm25_index.search(
            question,
            top_k=BM25_RETRIEVAL_K,
        )
        candidates = merge_candidates(
            candidates,
            bm25_candidates,
        )

    if use_reranker:
        return rerank_candidates(
            question,
            candidates,
            top_n=final_top_k,
        )

    return candidates[:final_top_k]
