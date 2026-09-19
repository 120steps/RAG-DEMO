import chromadb

from config import QUERY_REWRITE_ENABLED, TOP_K
from llm import generate_answer
from query_rewriter import rewrite_query
from retrieval import retrieve_candidates

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="company_knowledge")

def ask_rag(
    question: str,
    top_k: int = TOP_K,
    use_hybrid: bool = True,
    use_reranker: bool = True,
):
    original_query = question
    retrieval_query = (
        rewrite_query(original_query)
        if QUERY_REWRITE_ENABLED
        else original_query
    )

    candidates = retrieve_candidates(
        retrieval_query,
        collection,
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
        final_top_k=top_k,
    )

    documents = [candidate["document"] for candidate in candidates]
    metadatas = [candidate["metadata"] for candidate in candidates]
    distances = [
        candidate["original_distance"]
        for candidate in candidates
    ]

    context_parts = []

    for document, metadata, distance in zip(documents, metadatas, distances):
        context_parts.append(f"""
        Context:    {document}
        Source:    {metadata['source']}
        Chunk ID:    {metadata['chunk_id']}
        Page:    {metadata['page'] if 'page' in metadata else 'N/A'}
        """)

        # 调试使用
        # print(
        #     f"Retrieved document: {document}"
        #     f"\nRetrieved metadata: [chunk_id: {metadata['chunk_id']}, source: {metadata['source']}]"
        #     f"\nRetrieved distance: {distance}"
        # )

    context = "\n\n".join(context_parts)

    prompt = f"""
你是企业知识库助手。

智能根据下面的知识块回答问题。

要求：
1.不允许使用知识库之外的信息
2.如果知识库中没有答案，回答：
根据当前知识库无法回答该问题。
3. 回答必须简洁。
4. 回答最后必须注明引用来源
5. 引用格式：
   来源： 文件名，第X页，知识块ID：Y

知识库：
{context}

问题：
{original_query}
"""

    answer = generate_answer(prompt)
    return {
        "question": original_query,
        "original_query": original_query,
        "retrieval_query": retrieval_query,
        "answer": answer,
        "documents": documents,
        "metadatas": metadatas,
        "distances": distances,
        "bm25_scores": [
            candidate["bm25_score"]
            for candidate in candidates
        ],
        "rerank_scores": [
            candidate["rerank_score"]
            for candidate in candidates
        ],
    }
