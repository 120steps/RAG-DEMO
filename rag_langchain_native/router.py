"""Phase 9 安全 Query Router。

Router 只决定“闲聊”还是“知识库检索”，不是 Agent，也不能授予权限。不确定输入默认走
knowledge_rag，因此仍会经过 Authorized Retrieval；normal_chat 不读取任何 Context，
不会成为绕过 ACL 的侧门。
"""

from __future__ import annotations

import re

from langchain_core.runnables import RunnableLambda


NORMAL_CHAT = "normal_chat"
KNOWLEDGE_RAG = "knowledge_rag"


class QueryRouter:
    """用确定性小规则识别纯问候，其余问题安全地送入 RAG。"""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.runnable = RunnableLambda(self.route).with_config(
            run_name="enterprise_query_router"
        )

    def route(self, question: str) -> str:
        """返回 normal_chat 或 knowledge_rag；关闭 Router 时始终走 RAG。"""
        if not self.enabled:
            return KNOWLEDGE_RAG
        normalized = re.sub(r"[\s，。！？!?、]+", "", question).lower()
        greetings = {
            "你好",
            "您好",
            "嗨",
            "hello",
            "hi",
            "早上好",
            "下午好",
            "晚上好",
            "谢谢",
        }
        return NORMAL_CHAT if normalized in greetings else KNOWLEDGE_RAG

    @staticmethod
    def normal_chat_answer(question: str) -> str:
        """返回不读取知识库的固定礼貌响应，避免闲聊额外调用 Gemini。"""
        del question
        return "你好！我是企业知识库助手。请告诉我你想查询的制度或文档问题。"

