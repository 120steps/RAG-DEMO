from rag_service import ask_rag


# 1. 用户输入问题
question = input("Ask a question: ")

# 2. 调用RAG服务
result = ask_rag(question)

# 3. 输出结果
print("\n AI answer:")
print(result["answer"])

print("\n Retrieved documents:")
for i, (document, metadata, distance) in enumerate(
    zip(result["documents"], result["metadatas"], result["distances"]),
    start=1
    ):

    print()
    print(f"Top {i}")

    print(
        f"Source: {metadata['source']}, " 
        f"Chunk ID: {metadata['chunk_id']}, "
        f"Page: {metadata['page'] if 'page' in metadata else 'N/A'}, "
        f"Distance: {distance:.4f}"
    )

    print(f"Document {i}:" + document)