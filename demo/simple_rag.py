from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np

client = OpenAI()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 1. 读取文件
with open("data/company_policy.txt", "r", encoding="utf-8") as file:
    text = file.read()


# 2. chunk
chunks = [
    chunk.strip()
    for chunk in text.split("\n\n") 
    if chunk.strip()
]

# 3. embedding - OPEN AI API
# def get_embedding(context):
#     response = client.embeddings.create(
#         model="text-embedding-3-small",
#         input=context
#     )
#     return response.data[0].embedding

# 3. embedding - sentence-transformers
def get_embedding(context):
    return embedding_model.encode(context)  


# 4. 给所有知识块生成embedding
chunk_embeddings = []

for chunk in chunks:
    embedding = get_embedding(chunk)
    chunk_embeddings.append(embedding)

# 5. 用户输入问题
question = input("Ask a question: ")

#6. 获取问题的embedding
question_embedding = get_embedding(question)

# 7. 计算相似度
scores = []

for embedding in chunk_embeddings:
    score = np.dot(question_embedding, embedding)
    scores.append(score)

# 8. 找到最相似的知识块
best_index = np.argmax(scores)
best_chunk = chunks[best_index]

print("最相似的知识块:")
print(best_chunk)

# 9. 将最相似的知识块和问题一起发送给模型
prompt = f"根据以下知识块回答问题:\n\n企业制度: {best_chunk}\n\n问题: {question}\n\n如果企业制度中没有相关信息，请直接回答“抱歉，我无法回答这个问题。”"

response = client.responses.create(
    model="gpt-5.4-mini", 
    input=prompt
)

print("AI回答:")
print(response.output_text)