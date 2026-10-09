from __future__ import annotations

import pytest


def _create_admin(env, tenant="tenant-a"):
    return env["auth"].register_user(
        tenant_id=tenant,
        username=f"admin-{tenant}",
        password="password123",
        roles=["admin"],
    )


def _upload_index(env, document, content, filename="policy.pdf"):
    version = env["lifecycle"].upload_version(
        tenant_id=document["tenant_id"],
        knowledge_base_id=document["knowledge_base_id"],
        document_id=document["document_id"],
        filename=filename,
        content=content,
    )
    return env["lifecycle"].index_version(
        document["document_id"], version["version_id"]
    )


def test_document_version_publish_rollback_delete_restore_and_metadata(
    enterprise_env, make_pdf_bytes
):
    env = enterprise_env
    admin = _create_admin(env)
    lifecycle = env["lifecycle"]
    document = lifecycle.register_document(
        tenant_id="tenant-a",
        knowledge_base_id="default",
        document_name="Policy",
        source="policy.pdf",
        classification="general",
        owner=admin["user_id"],
    )
    # Register 同一 source 是幂等的，不会创建第二个逻辑文档。
    duplicate = lifecycle.register_document(
        tenant_id="tenant-a",
        knowledge_base_id="default",
        document_name="Policy",
        source="policy.pdf",
        classification="general",
        owner=admin["user_id"],
    )
    assert duplicate["document_id"] == document["document_id"]
    assert duplicate["created"] is False

    v1 = _upload_index(env, document, make_pdf_bytes("general policy version one"))
    lifecycle.publish_version(document["document_id"], v1["version_id"])
    assert env["catalog"].get_document(document["document_id"])["active_version_id"] == v1["version_id"]
    epoch = env["catalog"].knowledge_base_epoch("tenant-a", "default")
    lifecycle.publish_version(document["document_id"], v1["version_id"])
    assert env["catalog"].knowledge_base_epoch("tenant-a", "default") == epoch

    # 同内容重复上传返回同一版本。
    # 读取已保存的完全相同字节验证 content_hash 幂等。
    saved = lifecycle.download_path(document["document_id"]).read_bytes()
    same = lifecycle.upload_version(
        tenant_id="tenant-a",
        knowledge_base_id="default",
        document_id=document["document_id"],
        filename="policy.pdf",
        content=saved,
    )
    assert same["version_id"] == v1["version_id"]
    assert same["created"] is False

    v2 = _upload_index(env, document, make_pdf_bytes("general policy version two"))
    lifecycle.publish_version(document["document_id"], v2["version_id"])
    assert env["catalog"].get_version(v1["version_id"])["status"] == "retired"
    lifecycle.rollback_version(document["document_id"], v1["version_id"])
    assert env["catalog"].get_document(document["document_id"])["active_version_id"] == v1["version_id"]

    stored = env["store"].get(where={"version_id": v1["version_id"]}, include=["metadatas"])
    assert stored["metadatas"]
    metadata = stored["metadatas"][0]
    for key in (
        "tenant_id", "knowledge_base_id", "document_id", "version_id",
        "source", "page", "chunk_id", "chunk_uid",
    ):
        assert metadata.get(key) is not None

    lifecycle.soft_delete(document["document_id"])
    assert env["catalog"].active_documents("tenant-a", "default") == []
    lifecycle.restore(document["document_id"])
    assert env["catalog"].active_documents("tenant-a", "default")


def test_failed_ingestion_keeps_old_published_version(enterprise_env, make_pdf_bytes):
    env = enterprise_env
    admin = _create_admin(env)
    lifecycle = env["lifecycle"]
    document = lifecycle.register_document(
        tenant_id="tenant-a", knowledge_base_id="default", document_name="Policy",
        source="failure.pdf", classification="general", owner=admin["user_id"],
    )
    good = _upload_index(env, document, make_pdf_bytes("stable active content"), "failure.pdf")
    lifecycle.publish_version(document["document_id"], good["version_id"])
    # Phase 11：非法文件在创建版本前即拒绝，不能污染 Catalog/Chroma。
    with pytest.raises(ValueError, match="not a PDF"):
        lifecycle.upload_version(
            tenant_id="tenant-a", knowledge_base_id="default",
            document_id=document["document_id"], filename="failure.pdf", content=b"not a pdf",
        )
    assert len(env["catalog"].list_versions(document["document_id"])) == 1
    assert env["catalog"].get_document(document["document_id"])["active_version_id"] == good["version_id"]

