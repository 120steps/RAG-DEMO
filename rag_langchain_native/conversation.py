"""Phase 9 多轮会话存储与 Contextual Query Rewrite。

会话历史属于 tenant+user，Catalog 查询同时校验这两个字段，不能跨租户或读取其他用户。
历史只用于生成独立的 Retrieval Query；最终 Answer 仍回答用户本轮 Original Question。
Gemini 不可用时回退本轮原问题并标记 contextual_rewrite_failed，不会偷偷拼接历史事实。
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from .catalog import DocumentCatalog
from .config import DEFAULT_SETTINGS, Settings
from .observability import get_observability
from .security import Principal


CONTEXTUAL_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是企业知识库检索问题改写器。根据会话历史，把当前问题改写成可独立检索的"
            "一句问题。只解决指代和上下文缺失，不回答问题，不补充未经历史或当前问题"
            "证实的事实；保留数字、日期、名称、型号和缩写。只返回改写后的 Query。",
        ),
        ("human", "会话历史：\n{history}\n\n当前问题：{question}"),
    ]
)


class ConversationService:
    """提供会话 CRUD、消息持久化和上下文改写。"""

    def __init__(
        self,
        catalog: DocumentCatalog,
        model: Runnable | None = None,
        settings: Settings = DEFAULT_SETTINGS,
    ) -> None:
        self.catalog = catalog
        self.model = model
        self.settings = settings

    def create(self, principal: Principal, knowledge_base_id: str) -> dict:
        """创建只属于当前 verified Principal 的会话。"""
        self.catalog.ensure_knowledge_base(principal.tenant_id, knowledge_base_id)
        return self.catalog.create_conversation(
            principal.tenant_id, principal.user_id, knowledge_base_id
        )

    def get(self, principal: Principal, conversation_id: str) -> dict:
        """读取会话及最近历史；所有权不匹配时 Catalog 返回 Not Found。"""
        conversation = self.catalog.get_conversation(
            conversation_id, principal.tenant_id, principal.user_id
        )
        return {
            **conversation,
            "messages": self.catalog.list_messages(
                conversation_id, self.settings.conversation_history_limit
            ),
        }

    def delete(self, principal: Principal, conversation_id: str) -> None:
        self.catalog.delete_conversation(
            conversation_id, principal.tenant_id, principal.user_id
        )

    def add_message(
        self, principal: Principal, conversation_id: str, role: str, content: str
    ) -> dict:
        """先验证会话所有权，再保存消息。"""
        self.catalog.get_conversation(
            conversation_id, principal.tenant_id, principal.user_id
        )
        return self.catalog.add_message(conversation_id, role, content)

    def contextualize(
        self, principal: Principal, conversation_id: str | None, question: str
    ) -> tuple[str, bool]:
        """用授权历史把追问改写为独立 Retrieval Query。

        返回 ``(query, failed)``。无会话或空历史无需模型，直接返回原问题；模型缺失、空
        输出或异常都安全回退。History 限长避免 Prompt 无限增长和不必要成本。
        """
        original = question.strip()
        if not conversation_id:
            return original, False
        observability = get_observability()
        with observability.stage("conversation.load") as trace_data:
            self.catalog.get_conversation(
                conversation_id, principal.tenant_id, principal.user_id
            )
            messages = self.catalog.list_messages(
                conversation_id, self.settings.conversation_history_limit
            )
            trace_data["message_count"] = len(messages)
        if not messages:
            return original, False
        if self.model is None:
            return original, True
        history = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        try:
            # LCEL：dict -> ChatPromptValue -> AIMessage -> str。
            chain = CONTEXTUAL_REWRITE_PROMPT | self.model | StrOutputParser()
            with observability.stage("conversation.contextual_rewrite"):
                rewritten = chain.invoke(
                    {"history": history, "question": original},
                    config=observability.langchain_config(),
                ).strip()
            return (rewritten, False) if rewritten else (original, True)
        except Exception as error:
            observability.fallback("conversation.contextual_rewrite", error)
            return original, True

