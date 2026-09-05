from rag_service import ask_rag


# 1. 用户输入问题
question = input("Ask a question: ")

# 2. 调用RAG服务
result = ask_rag(question)

# 3. 输出结果
print("\n AI answer:")
print(result["answer"])

print("\n Retrieved documents:")
for metadata in result["metadatas"]:
    print(f"Source: {metadata['source']}, Chunk ID: {metadata['chunk_id']}")