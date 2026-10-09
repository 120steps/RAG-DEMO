from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag_langchain_native.enterprise import EnterpriseRAGService
from rag_langchain_native.security import (
    AuthenticationError,
    AuthorizationError,
    Principal,
    build_authorization_scope,
    require_document_access,
)


def _user(env, tenant, username, roles):
    value = env["auth"].register_user(
        tenant_id=tenant,
        username=username,
        password="password123",
        roles=roles,
    )
    token = env["auth"].login(tenant, username, "password123")
    return value, env["auth"].verify_token(token), token


def _published(env, *, tenant, source, classification, owner, content):
    lifecycle = env["lifecycle"]
    doc = lifecycle.register_document(
        tenant_id=tenant, knowledge_base_id="default", document_name=source,
        source=source, classification=classification, owner=owner,
    )
    version = lifecycle.upload_version(
        tenant_id=tenant, knowledge_base_id="default", document_id=doc["document_id"],
        filename=source, content=content,
    )
    version = lifecycle.index_version(doc["document_id"], version["version_id"])
    lifecycle.publish_version(doc["document_id"], version["version_id"])
    return doc, version


def test_signed_identity_rbac_acl_and_tamper_detection(enterprise_env, make_pdf_bytes):
    env = enterprise_env
    admin, admin_p, _ = _user(env, "tenant-a", "admin", ["admin"])
    _, general, token = _user(env, "tenant-a", "general", ["general"])
    _, hr, _ = _user(env, "tenant-a", "hr", ["hr"])
    _, finance, _ = _user(env, "tenant-a", "finance", ["finance"])
    general_doc, _ = _published(
        env, tenant="tenant-a", source="general.pdf", classification="general",
        owner=admin["user_id"], content=make_pdf_bytes("public handbook policy"),
    )
    hr_doc, _ = _published(
        env, tenant="tenant-a", source="hr.pdf", classification="hr",
        owner=admin["user_id"], content=make_pdf_bytes("secret human resource salary"),
    )
    finance_doc, _ = _published(
        env, tenant="tenant-a", source="finance.pdf", classification="finance",
        owner=admin["user_id"], content=make_pdf_bytes("secret finance budget"),
    )
    assert set(build_authorization_scope(env["catalog"], general, "default").document_ids) == {general_doc["document_id"]}
    assert hr_doc["document_id"] in build_authorization_scope(env["catalog"], hr, "default").document_ids
    assert finance_doc["document_id"] in build_authorization_scope(env["catalog"], finance, "default").document_ids
    assert len(build_authorization_scope(env["catalog"], admin_p, "default").document_ids) == 3
    with pytest.raises(AuthorizationError):
        require_document_access(env["catalog"], general, hr_doc["document_id"])
    env["catalog"].grant_acl(hr_doc["document_id"], "user", general.user_id)
    assert require_document_access(env["catalog"], general, hr_doc["document_id"])
    tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
    with pytest.raises(AuthenticationError):
        env["auth"].verify_token(tampered)


def test_group_acl_grants_only_the_named_group(enterprise_env, make_pdf_bytes):
    env = enterprise_env
    admin, _, _ = _user(env, "tenant-a", "admin", ["admin"])
    analyst = env["auth"].register_user(
        tenant_id="tenant-a", username="analyst", password="password123",
        roles=["general"], groups=["payroll-project"],
    )
    outsider, outside_principal, _ = _user(
        env, "tenant-a", "outsider", ["general"]
    )
    analyst_principal = env["auth"].verify_token(
        env["auth"].login("tenant-a", "analyst", "password123")
    )
    document, _ = _published(
        env, tenant="tenant-a", source="group.pdf", classification="hr",
        owner=admin["user_id"], content=make_pdf_bytes("group restricted payroll"),
    )
    env["catalog"].grant_acl(
        document["document_id"], "group", "payroll-project"
    )
    assert document["document_id"] in build_authorization_scope(
        env["catalog"], analyst_principal, "default"
    ).document_ids
    assert document["document_id"] not in build_authorization_scope(
        env["catalog"], outside_principal, "default"
    ).document_ids


def test_vector_bm25_fusion_and_citations_share_tenant_scope(enterprise_env, make_pdf_bytes):
    env = enterprise_env
    admin_a, principal_a, _ = _user(env, "tenant-a", "admin-a", ["admin"])
    admin_b, principal_b, _ = _user(env, "tenant-b", "admin-b", ["admin"])
    doc_a, version_a = _published(
        env, tenant="tenant-a", source="same.pdf", classification="general",
        owner=admin_a["user_id"], content=make_pdf_bytes("alpha tenant unique policy"),
    )
    doc_b, version_b = _published(
        env, tenant="tenant-b", source="same.pdf", classification="general",
        owner=admin_b["user_id"], content=make_pdf_bytes("beta tenant unique policy"),
    )
    service = EnterpriseRAGService(
        env["catalog"], env["settings"], vectorstore=env["store"]
    )
    result_a = service.retrieve_only(
        principal=principal_a, knowledge_base_id="default",
        question="alpha tenant unique policy", top_k=10,
    )
    result_b = service.retrieve_only(
        principal=principal_b, knowledge_base_id="default",
        question="beta tenant unique policy", top_k=10,
    )
    assert result_a["documents"] and result_b["documents"]
    assert {item["tenant_id"] for item in result_a["documents"]} == {"tenant-a"}
    assert {item["version_id"] for item in result_a["documents"]} == {version_a["version_id"]}
    assert {item["tenant_id"] for item in result_b["documents"]} == {"tenant-b"}
    assert doc_a["document_id"] != doc_b["document_id"]
    assert version_a["version_id"] != version_b["version_id"]


def test_empty_authorization_scope_never_falls_back_to_full_search(enterprise_env, make_pdf_bytes):
    env = enterprise_env
    admin, _, _ = _user(env, "tenant-a", "admin", ["admin"])
    _, general, _ = _user(env, "tenant-a", "general", ["general"])
    _published(
        env, tenant="tenant-a", source="hr-only.pdf", classification="hr",
        owner=admin["user_id"], content=make_pdf_bytes("highly confidential hr"),
    )
    service = EnterpriseRAGService(env["catalog"], env["settings"], vectorstore=env["store"])
    result = service.retrieve_only(
        principal=general, knowledge_base_id="default", question="confidential hr"
    )
    assert result["documents"] == []
    assert result["refusal_reason"] == "no_authorized_documents"


def test_client_metadata_filter_can_only_narrow_authorized_scope(enterprise_env, make_pdf_bytes):
    env = enterprise_env
    admin, _, _ = _user(env, "tenant-a", "admin", ["admin"])
    _, general, _ = _user(env, "tenant-a", "general", ["general"])
    general_doc, general_version = _published(
        env, tenant="tenant-a", source="general-filter.pdf", classification="general",
        owner=admin["user_id"], content=make_pdf_bytes("general visible filter"),
    )
    hr_doc, hr_version = _published(
        env, tenant="tenant-a", source="hr-filter.pdf", classification="hr",
        owner=admin["user_id"], content=make_pdf_bytes("restricted hr filter"),
    )
    allowed = build_authorization_scope(
        env["catalog"], general, "default", document_id=general_doc["document_id"]
    )
    assert allowed.version_ids == (general_version["version_id"],)
    blocked = build_authorization_scope(
        env["catalog"], general, "default",
        document_id=hr_doc["document_id"], version_id=hr_version["version_id"],
        classification="hr",
    )
    assert blocked.empty


def test_answer_context_and_citation_keep_authorized_version(enterprise_env, make_pdf_bytes):
    env = enterprise_env
    admin, principal, _ = _user(env, "tenant-a", "admin", ["admin"])
    _, version = _published(
        env, tenant="tenant-a", source="citation.pdf", classification="general",
        owner=admin["user_id"], content=make_pdf_bytes("approved enterprise citation"),
    )
    model = RunnableLambda(lambda _: AIMessage(content="approved enterprise citation"))
    service = EnterpriseRAGService(
        env["catalog"], env["settings"], vectorstore=env["store"], llm=model
    )
    result = service.chat(
        principal=principal, knowledge_base_id="default",
        question="approved enterprise citation",
    )
    assert result["sources"]
    assert {item["version_id"] for item in result["sources"]} == {version["version_id"]}
    assert {item["tenant_id"] for item in result["documents"]} == {"tenant-a"}

