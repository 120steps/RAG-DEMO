import json
from pathlib import Path
from threading import Lock

from google.genai import types

from config import QUERY_REWRITE_MODEL
from llm import client


CACHE_FILE = (
    Path(__file__).resolve().parent
    / "eval"
    / "query_rewrite_cache.json"
)
CACHE_LOCK = Lock()

QUERY_REWRITE_PROMPT = """你是企业知识库检索查询改写器。

请将用户问题改写成更适合企业知识库检索的一条查询，并严格遵守以下要求：
1. 保留用户的原始意图。
2. 不增加用户问题中没有提供的事实，也不要假设知识库内容。
3. 不回答问题。
4. 不解释改写过程。
5. 不改变数字、日期、人名、产品型号等关键实体。
6. 将口语化或模糊的表达改写成清晰、完整、适合检索的表达。
7. 只返回一条改写后的查询，不添加前缀、引号或其他内容。

用户问题：
{question}
"""


def _save_cache(cache):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = CACHE_FILE.with_suffix(".tmp")
    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_file.replace(CACHE_FILE)


def _load_cache():
    if not CACHE_FILE.exists():
        _save_cache({})
        return {}

    with CACHE_FILE.open("r", encoding="utf-8") as file:
        cache = json.load(file)

    if not isinstance(cache, dict):
        raise ValueError("Query rewrite cache must contain a JSON object")

    return cache


def rewrite_query(question: str) -> str:
    try:
        with CACHE_LOCK:
            cache = _load_cache()
            cached_query = cache.get(question)
            if isinstance(cached_query, str) and cached_query:
                return cached_query
    except (OSError, json.JSONDecodeError, ValueError):
        pass

    try:
        response = client.models.generate_content(
            model=QUERY_REWRITE_MODEL,
            contents=QUERY_REWRITE_PROMPT.format(question=question),
            config=types.GenerateContentConfig(
                temperature=0.0,
            ),
        )
        rewritten_query = response.text.strip()
    except Exception:
        return question

    if not rewritten_query:
        return question

    try:
        with CACHE_LOCK:
            cache = _load_cache()
            cache[question] = rewritten_query
            _save_cache(cache)
    except (OSError, json.JSONDecodeError, ValueError):
        pass

    return rewritten_query


if __name__ == "__main__":
    original_query = "国际出差是不是提前10天？"
    rewritten_query = rewrite_query(original_query)

    print(f"Original Query: {original_query}")
    print(f"Rewritten Query: {rewritten_query}")
