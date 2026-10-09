"""统一失败分类；业务拒答和正常无结果不属于系统异常。"""

from __future__ import annotations

from enum import StrEnum


class FailureType(StrEnum):
    AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
    AUTHORIZATION_FAILURE = "AUTHORIZATION_FAILURE"
    TENANT_ACCESS_FAILURE = "TENANT_ACCESS_FAILURE"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    DOCUMENT_VERSION_FAILURE = "DOCUMENT_VERSION_FAILURE"
    INGESTION_FAILURE = "INGESTION_FAILURE"
    EMBEDDING_FAILURE = "EMBEDDING_FAILURE"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    RERANKER_FAILURE = "RERANKER_FAILURE"
    LLM_RATE_LIMIT = "LLM_RATE_LIMIT"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_SERVICE_ERROR = "LLM_SERVICE_ERROR"
    CACHE_FAILURE = "CACHE_FAILURE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


def classify_exception(error: BaseException, operation: str = "") -> FailureType:
    """只根据异常类型/安全关键词分类，不把原始异常消息写进日志。"""
    from ..catalog import ConflictError, NotFoundError
    from ..security import AuthenticationError, AuthorizationError

    if isinstance(error, AuthenticationError):
        return FailureType.AUTHENTICATION_FAILURE
    if isinstance(error, AuthorizationError):
        return FailureType.AUTHORIZATION_FAILURE
    if isinstance(error, NotFoundError):
        return FailureType.DOCUMENT_NOT_FOUND
    if isinstance(error, ConflictError):
        return FailureType.DOCUMENT_VERSION_FAILURE
    name = type(error).__name__.lower()
    if "ratelimit" in name or "resourceexhausted" in name:
        return FailureType.LLM_RATE_LIMIT
    if isinstance(error, TimeoutError) or "timeout" in name:
        return FailureType.LLM_TIMEOUT
    operation = operation.lower()
    if "ingest" in operation or "index" in operation or "parse" in operation:
        return FailureType.INGESTION_FAILURE
    if "embed" in operation:
        return FailureType.EMBEDDING_FAILURE
    if "retriev" in operation or "vector" in operation or "bm25" in operation:
        return FailureType.RETRIEVAL_FAILURE
    if "rerank" in operation:
        return FailureType.RERANKER_FAILURE
    if "llm" in operation or "rewrite" in operation or "expand" in operation:
        return FailureType.LLM_SERVICE_ERROR
    return FailureType.INTERNAL_ERROR


def safe_error_summary(error: BaseException) -> str:
    """返回不会包含异常消息、文档内容或密钥的稳定摘要。"""
    return f"operation failed with {type(error).__name__}"

