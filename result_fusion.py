from bm25_search import candidate_key


RRF_K = 60


def fuse_results(result_sets, queries, rrf_k=RRF_K):
    if len(result_sets) != len(queries):
        raise ValueError("result_sets and queries must have equal lengths")
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")

    fused_by_key = {}
    first_seen_order = 0

    for query, candidates in zip(queries, result_sets):
        seen_in_query = set()

        for rank, candidate in enumerate(candidates, start=1):
            key = candidate_key(candidate)
            if key in seen_in_query:
                continue
            seen_in_query.add(key)

            if key not in fused_by_key:
                fused_by_key[key] = {
                    **candidate,
                    "distance": candidate.get("original_distance"),
                    "fusion_score": 0.0,
                    "matched_queries": [],
                    "query_ranks": {},
                    "_best_rank": rank,
                    "_first_seen_order": first_seen_order,
                }
                first_seen_order += 1

            fused = fused_by_key[key]
            fused["fusion_score"] += 1.0 / (rrf_k + rank)
            fused["matched_queries"].append(query)
            fused["query_ranks"][query] = rank
            fused["_best_rank"] = min(fused["_best_rank"], rank)

            distance = candidate.get("original_distance")
            current_distance = fused.get("original_distance")
            if distance is not None and (
                current_distance is None or distance < current_distance
            ):
                fused["original_distance"] = distance
                fused["distance"] = distance

            bm25_score = candidate.get("bm25_score")
            current_bm25_score = fused.get("bm25_score")
            if bm25_score is not None and (
                current_bm25_score is None
                or bm25_score > current_bm25_score
            ):
                fused["bm25_score"] = bm25_score

    fused_results = list(fused_by_key.values())
    fused_results.sort(
        key=lambda candidate: (
            -candidate["fusion_score"],
            candidate["_best_rank"],
            candidate["_first_seen_order"],
        )
    )

    for candidate in fused_results:
        candidate.pop("_best_rank")
        candidate.pop("_first_seen_order")

    return fused_results
