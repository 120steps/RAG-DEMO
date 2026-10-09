from dataclasses import replace

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag_langchain_native.query_processing import QueryProcessor


def test_query_rewrite_uses_lcel_and_cache(v3_settings):
    settings = replace(v3_settings, rewrite_enabled=True)
    model = RunnableLambda(lambda _: AIMessage(content="国际出差审批流程"))
    processor = QueryProcessor(model, settings)
    first = processor.process("国外出差领导怎么批？")
    second = QueryProcessor(None, settings).process("国外出差领导怎么批？")
    assert first["retrieval_query"] == "国际出差审批流程"
    assert second["retrieval_query"] == "国际出差审批流程"


def test_query_expansion_cache_and_failure_fallback(v3_settings):
    settings = replace(v3_settings, expansion_enabled=True)
    processor = QueryProcessor(None, settings)
    processor.expansion_cache.set(
        "MACsec支持吗？",
        ["MACsec支持吗？", "MACsec 功能支持范围", "MACsec 配置要求"],
    )
    cached = processor.process("MACsec支持吗？")
    assert len(cached["expanded_queries"]) == 3
    assert all("MACsec" in query for query in cached["expanded_queries"])

    failed = processor.process("没有缓存的问题")
    assert failed["expanded_queries"] == ["没有缓存的问题"]
    assert failed["expansion_failed"] is True


def test_enterprise_query_cache_namespace_isolated(v3_settings):
    settings = replace(v3_settings, rewrite_enabled=True)
    first = QueryProcessor(None, settings, cache_namespace="tenant-a:kb:1:user-a")
    second = QueryProcessor(None, settings, cache_namespace="tenant-b:kb:1:user-b")
    first.rewrite_cache.set(first._cache_key("相同问题"), "租户A改写")
    assert first.process("相同问题")["retrieval_query"] == "租户A改写"
    result = second.process("相同问题")
    assert result["retrieval_query"] == "相同问题"
    assert result["rewrite_failed"] is True

