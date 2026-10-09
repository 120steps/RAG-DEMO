"""Phase 9 文档生命周期：注册、上传版本、索引、发布、回滚、软删除和恢复。

Document 是长期稳定的逻辑对象；Document Version 是一次不可变的文件内容；Chunk 属于
某个具体 Version。更新文件不会覆盖旧版本：只有新版本完整写入 Chroma 并标记 indexed
后，Catalog 才允许把 active_version 指针切过去。

SQLite 与 Chroma 不是同一个事务系统。本实现采用“先索引、后发布”的可恢复策略：索引
失败把版本标记为 failed，并尽力删除该版本的残留 Chunk；旧 active version 始终不变。
发布/回滚只切 Catalog 指针并增加 epoch，旧向量可以保留，但 Authorized Retrieval 的
version filter 会立即排除它们。
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .catalog import ConflictError, DocumentCatalog
from .config import DEFAULT_SETTINGS, Settings
from .ingestion import load_pdf_pages, split_pages
from .vectorstore import add_documents, create_vectorstore


ALLOWED_CLASSIFICATIONS = {"general", "hr", "finance", "admin"}


def enterprise_settings(settings: Settings = DEFAULT_SETTINGS) -> Settings:
    """返回使用独立 Enterprise collection 的不可变 Settings 副本。"""
    return replace(settings, collection_name=settings.enterprise_collection_name)


def stable_chunk_uid(
    tenant_id: str,
    knowledge_base_id: str,
    document_id: str,
    version_id: str,
    page: int,
    chunk_id: int,
) -> str:
    """生成跨租户、文档、版本、页码都唯一且可重复计算的 Chunk ID。"""
    raw = "|".join(
        [
            tenant_id,
            knowledge_base_id,
            document_id,
            version_id,
            str(page),
            str(chunk_id),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DocumentLifecycleService:
    """协调 Catalog、文件系统、LangChain Loader/Splitter 与 Chroma。"""

    def __init__(
        self,
        catalog: DocumentCatalog,
        settings: Settings = DEFAULT_SETTINGS,
        vectorstore=None,
    ) -> None:
        self.catalog = catalog
        self.settings = enterprise_settings(settings)
        self.settings.ensure_runtime_dirs()
        self.vectorstore = vectorstore or create_vectorstore(self.settings)

    def register_document(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        document_name: str,
        source: str,
        document_type: str = "pdf",
        classification: str = "general",
        owner: str,
    ) -> dict[str, Any]:
        """注册逻辑文档，不上传内容。

        ``source`` 在租户+知识库内唯一；不同租户可以拥有同名 PDF。classification 决定
        默认 RBAC 边界，额外例外通过 ACL 配置。
        """
        value = classification.lower()
        if value not in ALLOWED_CLASSIFICATIONS:
            raise ValueError("Unsupported document classification")
        document, created = self.catalog.register_document(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            document_name=document_name,
            source=Path(source).name,
            document_type=document_type,
            classification=value,
            owner=owner,
        )
        return {**document, "created": created}

    def _version_path(
        self,
        tenant_id: str,
        knowledge_base_id: str,
        document_id: str,
        version_id: str,
        filename: str,
    ) -> Path:
        """构造严格位于 V3 runtime 的租户隔离文件路径。"""
        # 深层 tenant/kb/document/version 路径在 Windows 容易接近 MAX_PATH。原始文件名已
        # 保存在 Catalog source/document_name，因此磁盘采用固定短名，不影响 Citation。
        safe_name = "content.pdf" if Path(filename).suffix.lower() == ".pdf" else "content.bin"
        return (
            self.settings.upload_dir
            / "tenants"
            / tenant_id
            / "knowledge_bases"
            / knowledge_base_id
            / "documents"
            / document_id
            / "versions"
            / version_id
            / safe_name
        )

    def upload_version(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        document_id: str,
        filename: str,
        content: bytes,
    ) -> dict[str, Any]:
        """保存一个不可变版本；同内容重复上传返回已有版本。

        文件哈希用于幂等，而不是身份认证。路径中的 tenant/document/version 同时避免不同
        租户同名 PDF 冲突。
        """
        document = self.catalog.get_document(document_id)
        if (
            document["tenant_id"] != tenant_id
            or document["knowledge_base_id"] != knowledge_base_id
        ):
            raise ConflictError("Document does not belong to requested scope")
        if document["status"] == "deleted":
            raise ConflictError("Restore the document before uploading a version")
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported")
        content_hash = hashlib.sha256(content).hexdigest()
        existing = next(
            (
                version
                for version in self.catalog.list_versions(document_id)
                if version["content_hash"] == content_hash
            ),
            None,
        )
        if existing:
            return {**existing, "created": False}

        version_id = uuid.uuid4().hex
        destination = self._version_path(
            tenant_id,
            knowledge_base_id,
            document_id,
            version_id,
            filename,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".uploading")
        temporary.write_bytes(content)
        temporary.replace(destination)
        version, created = self.catalog.create_version(
            document_id=document_id,
            content_hash=content_hash,
            storage_path=str(destination),
            version_id=version_id,
        )
        return {**version, "created": created}

    def index_version(self, document_id: str, version_id: str) -> dict[str, Any]:
        """解析并索引指定版本，但不自动发布。

        Chunk Metadata 从这里开始完整携带 tenant/kb/document/version/source/page/chunk。
        发生异常时旧 active version 不受影响，新版本标记 failed，且尽力清理残留向量。
        """
        document = self.catalog.get_document(document_id)
        version = self.catalog.get_version(version_id)
        if version["document_id"] != document_id:
            raise ConflictError("Version does not belong to document")
        if version["status"] in {"indexed", "published", "retired"}:
            return {**version, "indexed": False, "idempotent": True}
        self.catalog.set_version_status(version_id, "indexing")
        try:
            pages = load_pdf_pages(version["storage_path"])
            chunks = split_pages(pages, self.settings)
            if not chunks:
                raise ValueError("PDF produced no chunks")
            enterprise_chunks: list[Document] = []
            for chunk in chunks:
                metadata = dict(chunk.metadata)
                page = int(metadata["page"])
                chunk_id = int(metadata["chunk_id"])
                uid = stable_chunk_uid(
                    document["tenant_id"],
                    document["knowledge_base_id"],
                    document_id,
                    version_id,
                    page,
                    chunk_id,
                )
                metadata.update(
                    tenant_id=document["tenant_id"],
                    knowledge_base_id=document["knowledge_base_id"],
                    document_id=document_id,
                    version_id=version_id,
                    document_name=document["document_name"],
                    source=document["source"],
                    document_type=document["document_type"],
                    classification=document["classification"],
                    chunk_uid=uid,
                )
                enterprise_chunks.append(
                    Document(
                        page_content=chunk.page_content,
                        metadata=metadata,
                        id=uid,
                    )
                )
            # 重试同一 version 时先清理该 version，范围由不可伪造的 Catalog ID 决定。
            self.vectorstore.delete(where={"version_id": version_id})
            ids = add_documents(self.vectorstore, enterprise_chunks)
            self.catalog.set_version_status(version_id, "indexed")
            return {
                **self.catalog.get_version(version_id),
                "indexed": True,
                "pages": len(pages),
                "chunks": len(ids),
                "chunk_ids": ids,
            }
        except Exception as error:
            try:
                self.vectorstore.delete(where={"version_id": version_id})
            finally:
                self.catalog.set_version_status(version_id, "failed", str(error))
            raise

    def publish_version(self, document_id: str, version_id: str) -> dict[str, Any]:
        """发布已成功索引的版本，使普通检索只看到它。"""
        return self.catalog.publish_version(document_id, version_id)

    def rollback_version(self, document_id: str, version_id: str) -> dict[str, Any]:
        """把历史 indexed/retired 版本重新设为 active，Citation 会随之切回该 version。"""
        return self.catalog.publish_version(document_id, version_id)

    def soft_delete(self, document_id: str) -> None:
        """只软删除目标文档；向量保留以便恢复，但授权 Scope 会立即排除。"""
        self.catalog.soft_delete_document(document_id)

    def restore(self, document_id: str) -> None:
        """恢复逻辑文档；其原 active version 重新进入授权 Scope。"""
        self.catalog.restore_document(document_id)

    def list_documents(
        self, tenant_id: str, knowledge_base_id: str, *, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        return self.catalog.list_documents(
            tenant_id, knowledge_base_id, include_deleted=include_deleted
        )

    def list_versions(self, document_id: str) -> list[dict[str, Any]]:
        return self.catalog.list_versions(document_id)

    def download_path(self, document_id: str, version_id: str | None = None) -> Path:
        """返回受控存储路径；调用方必须先执行 require_document_access。"""
        document = self.catalog.get_document(document_id)
        selected = version_id or document["active_version_id"]
        if not selected:
            raise ConflictError("Document has no active version")
        version = self.catalog.get_version(selected)
        if version["document_id"] != document_id:
            raise ConflictError("Version does not belong to document")
        path = Path(version["storage_path"]).resolve()
        root = self.settings.upload_dir.resolve()
        if not path.is_relative_to(root):
            raise ConflictError("Stored path escaped the V3 upload directory")
        return path

