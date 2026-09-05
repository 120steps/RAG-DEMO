import chromadb
from sentence_transformers import SentenceTransformer

# 1. 加载本地Ebedding模型
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. 连接本地Chroma数据库
client = chromadb.PersistentClient(path="./chroma_db")

# 3. 创建一个collection
collection = client.get_or_create_collection(name="company_policy")

# 4. 读取文件
with open("data/company_policy.txt", "r", encoding="utf-8") as file:
    text = file.read()

# 5. chunk
chunks = [
    chunk.strip()
    for chunk in text.split("\n\n")
    if chunk.strip()
]

metadatas = []

for i in range(len(chunks)):
    metadatas.append({
        "source": "company_policy",
        "chunk_id": f"chunk_{i}"
    })
    

# 6. 给所有知识块生成embedding
embeddings = embedding_model.encode(chunks).tolist()

# 7. 创建唯一id
ids = [f"chunk_{i}" for i in range(len(chunks))]

# 8. 将知识块和embedding存入Chroma数据库
collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings,
    metadatas=metadatas
)

print(f"成功写入{len(chunks)}条知识块到Chroma数据库。")
print(f"当前向量库共有 {collection.count()} 条知识块。")