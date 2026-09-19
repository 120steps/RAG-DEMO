from bm25_search import BM25Index, merge_candidates
from config import TOP_K
from embedding import embed_text
from reranker import bind_candidates, rerank_candidates
from result_fusion import fuse_results


VECTOR_RETRIEVAL_K = 10
BM25_RETRIEVAL_K = 10
RERANK_CANDIDATE_K = VECTOR_RETRIEVAL_K + BM25_RETRIEVAL_K


def _retrieve_candidate_pool(
    question,
    collection,
    use_hybrid,
    bm25_index,
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
        bm25_candidates = bm25_index.search(
            question,
            top_k=BM25_RETRIEVAL_K,
        )
        candidates = merge_candidates(
            candidates,
            bm25_candidates,
        )

    return [
        {
            **candidate,
            "distance": candidate.get("original_distance"),
            "fusion_score": None,
            "matched_queries": [question],
        }
        for candidate in candidates
    ]


def retrieve_candidates(
    question,
    collection,
    use_reranker=False,
    use_hybrid=False,
    bm25_index=None,
    final_top_k=TOP_K,
    expanded_queries=None,
):
    if use_hybrid:
        if bm25_index is None:
            bm25_index = BM25Index.from_chroma(collection)

    queries = expanded_queries or [question]
    queries = list(dict.fromkeys(queries))
    if not queries:
        queries = [question]
    elif question not in queries:
        queries.insert(0, question)

    if len(queries) == 1:
        candidates = _retrieve_candidate_pool(
            queries[0],
            collection,
            use_hybrid,
            bm25_index,
        )
    else:
        result_sets = [
            _retrieve_candidate_pool(
                query,
                collection,
                use_hybrid,
                bm25_index,
            )
            for query in queries
        ]
        candidates = fuse_results(result_sets, queries)
        candidates = candidates[:RERANK_CANDIDATE_K]

    if use_reranker:
        return rerank_candidates(
            question,
            candidates,
            top_n=final_top_k,
        )

    return candidates[:final_top_k]
