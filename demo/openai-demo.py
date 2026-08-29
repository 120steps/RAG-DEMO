from openai import OpenAI

client = OpenAI()

question = input("Ask a question: ")

response = client.responses.create(
    # model="gpt-5.6-luna",
    model="gpt-5.4-mini",
    input=question
)

print(response.output_text)