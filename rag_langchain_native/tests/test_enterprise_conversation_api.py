from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag_langchain_native.api import create_app
from rag_langchain_native.catalog import NotFoundError
from rag_langchain_native.conversation import ConversationService
from rag_langchain_native.router import KNOWLEDGE_RAG, NORMAL_CHAT, QueryRouter
from rag_langchain_native.security import Principal


def _register(auth, tenant, username, roles):
    user = auth.register_user(
        tenant_id=tenant, username=username, password="password123", roles=roles
    )
    principal = auth.verify_token(auth.login(tenant, username, "password123"))
    return user, principal


def test_conversation_contextual_rewrite_history_limit_and_ownership(enterprise_env):
    env = enterprise_env
    user, principal = _register(env["auth"], "tenant-a", "alice", ["general"])
    _, other = _register(env["auth"], "tenant-a", "bob", ["general"])
    model = RunnableLambda(
        lambda _: AIMessage(content="国际出差申请最终由谁批准？")
    )
    service = ConversationService(
        env["catalog"], model=model, settings=env["settings"]
    )
    conversation = service.create(principal, "default")
    # 空历史不需要调用模型。
    assert service.contextualize(principal, conversation["conversation_id"], "第一次问题") == ("第一次问题", False)
    service.add_message(
        principal,
        conversation["conversation_id"],
        "user",
        "国际出差提前多久申请？",
    )
    rewritten, failed = service.contextualize(
        principal, conversation["conversation_id"], "那谁批准？"
    )
    assert rewritten == "国际出差申请最终由谁批准？"
    assert failed is False
    with pytest.raises(NotFoundError):
        service.get(other, conversation["conversation_id"])
    for index in range(env["settings"].conversation_history_limit + 3):
        service.add_message(principal, conversation["conversation_id"], "user", str(index))
    assert len(service.get(principal, conversation["conversation_id"])["messages"]) == env["settings"].conversation_history_limit
    service.delete(principal, conversation["conversation_id"])
    with pytest.raises(NotFoundError):
        service.get(principal, conversation["conversation_id"])


def test_router_is_safe_default_and_not_an_agent():
    router = QueryRouter(enabled=True)
    assert router.runnable.invoke("你好！") == NORMAL_CHAT
    assert router.runnable.invoke("公司差旅政策是什么？") == KNOWLEDGE_RAG
    assert router.runnable.invoke("不确定的复杂输入 xyz") == KNOWLEDGE_RAG
    assert QueryRouter(enabled=False).route("你好") == KNOWLEDGE_RAG


def test_api_auth_management_conversation_and_download(
    fake_store, v3_settings, make_pdf_bytes
):
    model = RunnableLambda(lambda _: AIMessage(content="测试回答"))
    app = create_app(v3_settings, vectorstore=fake_store, llm=model)
    auth = app.state.auth_service
    _register(auth, "tenant-a", "admin", ["admin"])
    _register(auth, "tenant-a", "employee", ["general"])
    _register(auth, "tenant-b", "other-admin", ["admin"])
    app.state.catalog.ensure_knowledge_base("tenant-a", "default")
    app.state.catalog.ensure_knowledge_base("tenant-b", "default")
    client = TestClient(app)

    def login(tenant, username):
        response = client.post(
            "/auth/login",
            json={"tenant_id": tenant, "username": username, "password": "password123"},
        )
        assert response.status_code == 200
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    admin_headers = login("tenant-a", "admin")
    user_headers = login("tenant-a", "employee")
    other_headers = login("tenant-b", "other-admin")
    assert client.post("/retrieve", json={"question": "x"}).status_code == 401
    # 客户端伪造 role/tenant 字段不会改变经过验证的 Principal。
    response = client.post(
        "/retrieve",
        json={"question": "x", "role": "admin", "tenant_id": "tenant-b"},
        headers=user_headers,
    )
    assert response.status_code == 200
    assert response.json()["tenant_id"] == "tenant-a"
    assert client.post(
        "/upload",
        files={"file": ("x.pdf", make_pdf_bytes("x"), "application/pdf")},
        headers=user_headers,
    ).status_code == 403

    response = client.post(
        "/upload",
        files={
            "file": (
                "enterprise.pdf",
                make_pdf_bytes("enterprise general policy"),
                "application/pdf",
            )
        },
        data={"knowledge_base_id": "default", "classification": "general"},
        headers=admin_headers,
    )
    assert response.status_code == 200, response.text
    document = response.json()["document"]
    version = response.json()["version"]
    assert client.post(
        f"/documents/{document['document_id']}/publish",
        json={"version_id": version["version_id"]},
        headers=admin_headers,
    ).status_code == 200
    assert client.get(
        f"/documents/{document['document_id']}/download",
        headers=user_headers,
    ).status_code == 200
    assert client.get(
        f"/documents/{document['document_id']}/download",
        headers=other_headers,
    ).status_code == 403

    conversation = client.post(
        "/conversations", json={"knowledge_base_id": "default"}, headers=user_headers
    )
    assert conversation.status_code == 200
    conversation_id = conversation.json()["conversation_id"]
    chat = client.post(
        "/chat",
        json={
            "question": "你好",
            "knowledge_base_id": "default",
            "conversation_id": conversation_id,
        },
        headers=user_headers,
    )
    assert chat.status_code == 200
    assert chat.json()["route"] == NORMAL_CHAT
    history = client.get(
        f"/conversations/{conversation_id}", headers=user_headers
    ).json()["messages"]
    assert [item["role"] for item in history] == ["user", "assistant"]

