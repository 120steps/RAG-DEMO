import os
import chromadb
from sentence_transformers import SentenceTransformer
from document_loader import load_pdf

DATA_DIR = "./data/txt"
PDF_DIR = "./data/pdf"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "company_knowledge"

# 1. 加载本地Ebedding模型
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. 连接本地Chroma数据库
client = chromadb.PersistentClient(path=CHROMA_DIR)

# 3. 创建一个collection
try:
    client.delete_collection(
        name = COLLECTION_NAME
    )
except Exception:
    pass

collection = client.get_or_create_collection(name=COLLECTION_NAME)

all_chunks = []
all_metadatas = []
all_ids = []

# 4.1 读取pdf文件
for filename in os.listdir(PDF_DIR):
    if not filename.endswith(".pdf"):
        continue

    file_path = os.path.join(PDF_DIR, filename)

    print(f"正在处理文件: {filename}")

    docs = load_pdf(file_path)

    for doc in docs:
        document_id = (
            f"{doc['source']}"
            f"_page_{doc['page']}"
            f"_chunk_{doc['chunk_index']}"
        )
        metadata = {
            "source": doc["source"],
            "chunk_id": doc["chunk_index"],
            "page": doc["page"]
        }
        all_chunks.append(doc["text"])
        all_metadatas.append(metadata)
        all_ids.append(document_id)

# 4.2 读取txt文件
for filename in os.listdir(DATA_DIR):
    if not filename.endswith(".txt"):
        continue

    file_path = os.path.join(DATA_DIR, filename)

    print(f"正在处理文件: {filename}")

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # 5. chunk
    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    # 6. 为每个chunk创建metadata和唯一id
    for chunk_index, chunk in enumerate(chunks):
        chunk_id = f"{filename}_chunk_{chunk_index}"
        metadata = {
            "source": filename,
            "chunk_id": chunk_id
        }
        all_chunks.append(chunk)
        all_metadatas.append(metadata)
        all_ids.append(chunk_id)

# 7. 给所有知识块生成embedding
embeddings = embedding_model.encode(all_chunks).tolist()

# 8. 将知识块和embedding存入Chroma数据库
collection.upsert(
    ids=all_ids,
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=all_metadatas
)

print()
print("知识库入库完成")
print(
    f"文件数量："
    f"{len(set(m['source'] for m in all_metadatas))}"
)
print(
    f"知识块数量：{len(all_chunks)}"
)
print(
    f"Chroma数据量：{collection.count()}条"
)