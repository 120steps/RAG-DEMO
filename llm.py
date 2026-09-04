from google import genai
from config import GEMINI_API_KEY, LLM_MODEL

client = genai.Client(api_key=GEMINI_API_KEY)

def generate_answer(prompt: str) -> str:
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )
    return response.text

# 测试llm连通性
# if __name__ == "__main__":
#     test_prompt = "请用中文简要介绍一下人工智能的概念。"
#     answer = generate_answer(test_prompt)
#     print("模型回答:")
#     print(answer)