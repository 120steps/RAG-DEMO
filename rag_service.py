import chromadb

from sentence_transformers import SentenceTransformer
from llm import generate_answer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="company_knowledge")

def ask_rag(
    question: str,
    top_k: int = 3
):
    question_embedding = (
        embedding_model.encode(question).tolist()   
    )

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []

    for document, metadata, distance in zip(documents, metadatas, distances):
        context_parts.append(f"""
        context:    {document}
        source:    {metadata['source']}
        Chunk ID:    {metadata['chunk_id']}
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

如果知识库中没有答案，
回答：
根据当前知识库无法回答该问题。

知识库：
{context}

问题：
{question}
"""

    answer = generate_answer(prompt)
    return {
        "question": question,
        "answer": answer,
        "documents": documents,
        "metadatas": metadatas,
        "distances": distances
    }