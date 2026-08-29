import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# 1. openai client
load_dotenv()  # Load environment variables from .env file
client = OpenAI()

# 2. 加载本地Ebedding模型
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. 连接本地Chroma数据库
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 4. 获取知识库collection
collection = chroma_client.get_or_create_collection(name="company_policy")

# 5. 用户输入问题
question = input("Ask a question: ")

# 6. 获取问题的embedding
question_embedding = embedding_model.encode(question).tolist()

# 7. 在Chroma数据库中查找最相似的3个知识块
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

# 8. 获取检索结果
retrieved_chunks = results["documents"][0]

print("\n检索到的知识块:")

for index, chunk in enumerate(
    retrieved_chunks,
    start=1
):
    print(f"\n----  Chunk {index}. {chunk}")

# 9. 将检索到的知识块和问题一起发送给模型
context = "\n\n".join(retrieved_chunks)

# 10.构建prompt
prompt = f"根据以下知识块回答问题:\n\n企业制度: {context}\n\n问题: {question}\n\n如果企业制度中没有相关信息，请直接回答“抱歉，我无法回答这个问题。”"
print("\nPrompt:")
print(prompt)

# 11. 调用OpenAI API获取回答
response = client.responses.create(
    model="gpt-5.4-mini",
    input=prompt
)

print("\n模型回答:")
print(response.output_text) 