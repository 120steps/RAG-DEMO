"""Phase 9 Evaluation 的显式租户、授权上下文与真实 PDF bootstrap。"""

from __future__ import annotations

from pathlib import Path

from ..catalog import DocumentCatalog
from ..config import DEFAULT_SETTINGS, Settings
from ..lifecycle import DocumentLifecycleService
from ..security import Principal, hash_password


EVAL_TENANT = "evaluation-tenant"
EVAL_KNOWLEDGE_BASE = "phase9-ground-truth"
EVAL_USERNAME = "evaluation-admin"


def evaluation_principal(
    catalog: DocumentCatalog,
) -> Principal:
    """创建/加载服务端 Catalog 中的评估管理员，不通过关闭权限绕过授权。"""
    catalog.ensure_knowledge_base(EVAL_TENANT, EVAL_KNOWLEDGE_BASE)
    user = catalog.find_user(EVAL_TENANT, EVAL_USERNAME)
    if user is None:
        user = catalog.create_user(
            tenant_id=EVAL_TENANT,
            username=EVAL_USERNAME,
            password_hash=hash_password("evaluation-only-password"),
            roles=["admin"],
        )
    return Principal(
        user_id=user["user_id"],
        tenant_id=user["tenant_id"],
        roles=tuple(user["roles"]),
        groups=tuple(user["groups"]),
    )


def bootstrap_evaluation_knowledge_base(
    settings: Settings = DEFAULT_SETTINGS,
) -> dict:
    """把现有真实 PDF 幂等建入 Phase 9 独立 collection 并发布。"""
    catalog = DocumentCatalog(settings.catalog_path)
    principal = evaluation_principal(catalog)
    lifecycle = DocumentLifecycleService(catalog, settings)
    details = []
    for path in sorted(Path(settings.source_pdf_dir).glob("*.pdf")):
        document = lifecycle.register_document(
            tenant_id=principal.tenant_id,
            knowledge_base_id=EVAL_KNOWLEDGE_BASE,
            document_name=path.name,
            source=path.name,
            classification="general",
            owner=principal.user_id,
        )
        if document["status"] == "deleted":
            lifecycle.restore(document["document_id"])
        version = lifecycle.upload_version(
            tenant_id=principal.tenant_id,
            knowledge_base_id=EVAL_KNOWLEDGE_BASE,
            document_id=document["document_id"],
            filename=path.name,
            content=path.read_bytes(),
        )
        if version["status"] in {"registered", "failed"}:
            version = lifecycle.index_version(
                document["document_id"], version["version_id"]
            )
        lifecycle.publish_version(document["document_id"], version["version_id"])
        details.append(
            {
                "source": path.name,
                "document_id": document["document_id"],
                "version_id": version["version_id"],
                "status": "published",
            }
        )
    return {
        "tenant_id": EVAL_TENANT,
        "knowledge_base_id": EVAL_KNOWLEDGE_BASE,
        "principal": principal,
        "catalog": catalog,
        "documents": details,
    }

