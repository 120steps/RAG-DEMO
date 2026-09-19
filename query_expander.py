import json
import re
import time
from pathlib import Path
from threading import Lock

from google.genai import types

from config import QUERY_EXPANSION_COUNT, QUERY_EXPANSION_MODEL
from llm import client


CACHE_FILE = (
    Path(__file__).resolve().parent
    / "eval"
    / "query_expansion_cache.json"
)
CACHE_LOCK = Lock()
ENTITY_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[A-Za-z0-9][A-Za-z0-9._-]*"
    r"(?![A-Za-z0-9])"
)

QUERY_EXPANSION_PROMPT = """你是企业知识库检索 Query Expansion 工具。

请为输入查询生成恰好 {expansion_count} 条不同的检索查询，并严格遵守：
1. 保留输入查询的原始意图。
2. 不补充输入中没有的事实，也不假设知识库内容。
3. 不回答问题，不推断答案，不解释生成过程。
4. 数字、日期、协议名、产品型号、缩写、人名、组织名等关键实体必须原样保留。
5. MACsec、BGP、OSPF、NAC、802.1X 等技术术语不得改写或省略。
6. 每条查询从不同检索角度表达同一需求，不得语义重复。
7. 不要重复输入查询本身。
8. 只返回 JSON 字符串数组，不返回其他内容。

输入查询：
{query}
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
        raise ValueError("Query expansion cache must contain a JSON object")

    return cache


def _protected_entities(query):
    entities = []
    for token in ENTITY_TOKEN_PATTERN.findall(query):
        uppercase_count = sum(character.isupper() for character in token)
        if any(character.isdigit() for character in token) or uppercase_count >= 2:
            entities.append(token)
    return entities


def _normalize_queries(query, expansions):
    if not isinstance(expansions, list):
        return [query]

    protected_entities = _protected_entities(query)
    queries = [query]
    seen = {query}

    for expansion in expansions:
        if not isinstance(expansion, str):
            continue

        expansion = expansion.strip()
        if not expansion or expansion in seen:
            continue
        if any(entity not in expansion for entity in protected_entities):
            continue

        queries.append(expansion)
        seen.add(expansion)
        if len(queries) == QUERY_EXPANSION_COUNT + 1:
            break

    return queries


def get_cached_expansion(query):
    try:
        with CACHE_LOCK:
            cache = _load_cache()
            cached_queries = cache.get(query)
    except (OSError, json.JSONDecodeError, ValueError):
        return None

    if not isinstance(cached_queries, list):
        return None

    normalized = _normalize_queries(query, cached_queries)
    return (
        normalized
        if len(normalized) == QUERY_EXPANSION_COUNT + 1
        else None
    )


def expand_query_with_status(query: str):
    cached_queries = get_cached_expansion(query)
    if cached_queries is not None:
        return cached_queries, False

    try:
        response = client.models.generate_content(
            model=QUERY_EXPANSION_MODEL,
            contents=QUERY_EXPANSION_PROMPT.format(
                expansion_count=QUERY_EXPANSION_COUNT,
                query=query,
            ),
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=list[str],
            ),
        )
        expansions = response.parsed
        if not isinstance(expansions, list):
            expansions = json.loads(response.text)
        expanded_queries = _normalize_queries(query, expansions)
    except Exception:
        return [query], True

    if len(expanded_queries) != QUERY_EXPANSION_COUNT + 1:
        return [query], True

    try:
        with CACHE_LOCK:
            cache = _load_cache()
            cache[query] = expanded_queries
            _save_cache(cache)
    except (OSError, json.JSONDecodeError, ValueError):
        pass

    return expanded_queries, False


def expand_query(query: str) -> list[str]:
    expanded_queries, _ = expand_query_with_status(query)
    return expanded_queries


if __name__ == "__main__":
    test_queries = (
        "国外出差领导怎么批？",
        "MACsec支持吗？",
        "密码多久换？",
    )

    for index, original_query in enumerate(test_queries):
        if index:
            time.sleep(3.5)
        expanded_queries, expansion_failed = expand_query_with_status(
            original_query
        )
        print(f"Original Query: {original_query}")
        print(f"Expanded Queries: {expanded_queries}")
        print(f"Expansion Failed: {expansion_failed}")
        print()
