import chromadb
from sentence_transformers import SentenceTransformer
from llm import generate_answer

# 1. 加载本地Ebedding模型
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. 连接本地Chroma数据库
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 3. 获取知识库collection
collection = chroma_client.get_or_create_collection(name="company_policy")

# 4. 用户输入问题
question = input("Ask a question: ")

# 5. 获取问题的embedding
question_embedding = embedding_model.encode(question).tolist()

# 6. 在Chroma数据库中查找最相似的3个知识块
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

# 7. 获取检索结果
retrieved_chunks = results["documents"][0]
retrieved_metadatas = results["metadatas"][0]
retrieved_distances = results["distances"][0]

print("\n检索到的知识块:")

for index, (chunk, metadata, distance) in enumerate(
    zip(retrieved_chunks, retrieved_metadatas, retrieved_distances),
    start=1
):
    print(f"\n----  Chunk {index}. {chunk}")
    print(
        f"Metadata: {metadata['source']}"
        f"/ Chunk {metadata['chunk_id']}"
        f"/ Distance: {distance}"
        )

# 8. 将检索到的知识块和问题一起发送给模型
context_parts = []

for chunk, metadata in zip(retrieved_chunks, retrieved_metadatas):
    context_parts.append(f"""
    context:    {chunk}
    source:    {metadata['source']}
    Chunk ID:    {metadata['chunk_id']}
    """)

context = "\n\n".join(context_parts)

# 9.构建prompt
prompt = f"""
你是企业知识库助手。
只能根据以下知识块回答问题。 
回答要求：
1.不允许使用知识库之外的信息。 
2.如果知识库没有答案，请回答：根据当前知识库无法回答。 
3.回答最后需要列出引用来源。  

知识库: {context}

问题: {question}
"""

print("\nPrompt:")
print(prompt)

# 10. 调用API获取回答
answer = generate_answer(prompt)


print("\n模型回答:")
print(answer) 