"""Phase 9 Enterprise RAG 应用服务：把认证后的上下文接入 V3 LCEL Pipeline。

本模块不重新实现 Vector/BM25/RRF/Reranker，而是为现有 ``NativeRAGService`` 提供由
Catalog 计算的 Metadata Filter、active version 集合和 tenant-aware Cache Namespace。
执行顺序是：会话上下文改写 -> Router -> Authorized Retrieval -> 原 LCEL RAG ->
Citation 二次校验 -> 会话存储。
"""

from __future__ import annotations

import time
import warnings
from typing import Any

from langchain_core.runnables import Runnable

from .catalog import ConflictError, DocumentCatalog
from .chain import NativeRAGService, get_chat_model
from .config import DEFAULT_SETTINGS, Settings
from .conversation import ConversationService
from .lifecycle import enterprise_settings
from .router import KNOWLEDGE_RAG, QueryRouter
from .security import (
    AuthorizationError,
    AuthorizationScope,
    Principal,
    build_authorization_scope,
)
from .vectorstore import create_vectorstore


class EnterpriseRAGService:
    """权限感知的 V3 RAG 门面，供 FastAPI 与企业 Evaluation 共用。"""

    def __init__(
        self,
        catalog: DocumentCatalog,
        settings: Settings = DEFAULT_SETTINGS,
        *,
        vectorstore=None,
        llm: Runnable | None = None,
        query_model: Runnable | None = None,
        reranker=None,
    ) -> None:
        self.catalog = catalog
        self.settings = enterprise_settings(settings)
        self.vectorstore = vectorstore or create_vectorstore(self.settings)
        self._llm = llm
        self._query_model = query_model
        self._reranker = reranker
        self.router = QueryRouter(settings.query_router_enabled)
        self.conversations = ConversationService(
            catalog,
            model=query_model or llm,
            settings=settings,
        )
        # Scope+epoch 作为 key；发布/回滚后 epoch 改变，自然不会复用旧 BM25/Chain。
        self._services: dict[str, NativeRAGService] = {}

    def _scope(
        self,
        principal: Principal,
        knowledge_base_id: str,
        *,
        document_id: str | None = None,
        version_id: str | None = None,
        classification: str | None = None,
    ) -> AuthorizationScope:
        return build_authorization_scope(
            self.catalog,
            principal,
            knowledge_base_id,
            document_id=document_id,
            version_id=version_id,
            classification=classification,
        )

    def _service_for(
        self, principal: Principal, scope: AuthorizationScope
    ) -> NativeRAGService:
        permission_fingerprint = ",".join(
            [principal.user_id, *principal.roles, *principal.groups]
        )
        key = f"{scope.cache_namespace}:{permission_fingerprint}"
        service = self._services.get(key)
        if service is None:
            service = NativeRAGService(
                settings=self.settings,
                vectorstore=self.vectorstore,
                llm=self._llm,
                query_model=self._query_model,
                reranker=self._reranker,
                metadata_filter=scope.metadata_filter(),
                allowed_version_ids=set(scope.version_ids),
                cache_namespace=key,
            )
            self._services[key] = service
            # Prototype 进程内最多保留少量旧 epoch Service，避免无界增长。
            if len(self._services) > 64:
                oldest = next(iter(self._services))
                if oldest != key:
                    self._services.pop(oldest, None)
        return service

    @staticmethod
    def _empty_authorized_result(
        question: str, retrieval_query: str, reason: str
    ) -> dict[str, Any]:
        """无可见文档时 fail closed，且不构建/调用 Gemini。"""
        return {
            "question": question,
            "original_query": question,
            "retrieval_query": retrieval_query,
            "expanded_queries": [retrieval_query],
            "rewrite_failed": False,
            "expansion_failed": False,
            "answer": "根据当前授权知识库无法回答该问题。",
            "refused": True,
            "refusal_reason": reason,
            "sources": [],
            "documents": [],
            "retrieved_context": "",
            "retrieval_latency_ms": 0.0,
        }

    @staticmethod
    def _verify_output_scope(result: dict, scope: AuthorizationScope) -> None:
        """防御性检查：任何返回 Document/Citation 都必须属于授权 active version。"""
        allowed = set(scope.version_ids)
        for item in [*result.get("documents", []), *result.get("sources", [])]:
            version = item.get("version_id")
            if version not in allowed:
                raise AuthorizationError("Unauthorized version reached RAG output")
            if item.get("tenant_id") not in {None, scope.tenant_id}:
                raise AuthorizationError("Cross-tenant result blocked")

    def _contextual_query(
        self,
        principal: Principal,
        conversation_id: str | None,
        question: str,
    ) -> tuple[str, bool]:
        # 真实环境没有注入模型时，仅在确实存在历史并需要改写时创建 Gemini。
        if conversation_id and self.conversations.model is None:
            try:
                self.conversations.model = get_chat_model(self.settings)
            except Exception as error:
                # 只记录异常类型，避免异常正文意外包含 Prompt、Token 或文档内容。
                warnings.warn(
                    f"Contextual rewrite unavailable ({type(error).__name__}); using original question",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return self.conversations.contextualize(
            principal, conversation_id, question
        )

    def retrieve_only(
        self,
        *,
        principal: Principal,
        knowledge_base_id: str,
        question: str,
        top_k: int = 10,
        conversation_id: str | None = None,
        document_id: str | None = None,
        version_id: str | None = None,
        classification: str | None = None,
    ) -> dict[str, Any]:
        """执行带版本/ACL/租户边界的 Retrieval-only，不生成 Answer。"""
        contextual, failed = self._contextual_query(
            principal, conversation_id, question
        )
        scope = self._scope(
            principal,
            knowledge_base_id,
            document_id=document_id,
            version_id=version_id,
            classification=classification,
        )
        if scope.empty:
            result = self._empty_authorized_result(
                question, contextual, "no_authorized_documents"
            )
            result.pop("answer")
        else:
            result = self._service_for(principal, scope).retrieve_only(
                question,
                retrieval_question=contextual,
                top_k=top_k,
            )
            self._verify_output_scope(result, scope)
        result.update(
            contextual_query=contextual,
            contextual_rewrite_failed=failed,
            tenant_id=principal.tenant_id,
            knowledge_base_id=knowledge_base_id,
            authorization_epoch=scope.epoch,
        )
        return result

    def chat(
        self,
        *,
        principal: Principal,
        knowledge_base_id: str,
        question: str,
        conversation_id: str | None = None,
        document_id: str | None = None,
        version_id: str | None = None,
        classification: str | None = None,
    ) -> dict[str, Any]:
        """执行企业问答并把本轮 user/assistant 消息写入授权会话。"""
        if conversation_id:
            conversation = self.catalog.get_conversation(
                conversation_id, principal.tenant_id, principal.user_id
            )
            if conversation["knowledge_base_id"] != knowledge_base_id:
                raise ConflictError("Conversation belongs to another knowledge base")
        contextual, contextual_failed = self._contextual_query(
            principal, conversation_id, question
        )
        route = self.router.runnable.invoke(question)
        started = time.perf_counter()
        if route != KNOWLEDGE_RAG:
            result = {
                "question": question,
                "original_query": question,
                "retrieval_query": contextual,
                "expanded_queries": [contextual],
                "answer": self.router.normal_chat_answer(question),
                "refused": False,
                "refusal_reason": "normal_chat",
                "sources": [],
                "documents": [],
                "retrieved_context": "",
                "retrieval_latency_ms": 0.0,
            }
            scope = None
        else:
            scope = self._scope(
                principal,
                knowledge_base_id,
                document_id=document_id,
                version_id=version_id,
                classification=classification,
            )
            if scope.empty:
                result = self._empty_authorized_result(
                    question, contextual, "no_authorized_documents"
                )
            else:
                result = self._service_for(principal, scope).ask_rag(
                    question, retrieval_question=contextual
                )
                self._verify_output_scope(result, scope)
        result.update(
            route=route,
            contextual_query=contextual,
            contextual_rewrite_failed=contextual_failed,
            tenant_id=principal.tenant_id,
            knowledge_base_id=knowledge_base_id,
            authorization_epoch=scope.epoch if scope else None,
            total_latency_ms=(time.perf_counter() - started) * 1000,
        )
        if conversation_id:
            self.conversations.add_message(
                principal, conversation_id, "user", question
            )
            self.conversations.add_message(
                principal, conversation_id, "assistant", result["answer"]
            )
        return result

    def invalidate_services(self) -> None:
        """管理操作后可主动释放旧内存 BM25；epoch 本身已提供正确性隔离。"""
        self._services.clear()

