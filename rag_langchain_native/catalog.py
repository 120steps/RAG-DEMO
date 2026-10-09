"""Phase 9 企业文档目录、版本、ACL、用户与会话的 SQLite 持久层。

SQLite 与 Chroma 的职责不同：Catalog 保存“谁可以访问什么、哪个版本有效、文件位于
哪里”等强关系数据；Chroma 保存 Chunk 文本与向量，负责相似度检索。权限和版本关系
不应只塞进 Chroma Metadata，因为发布、回滚和软删除需要一致、可审计的状态转换。

本模块不执行 Embedding 或 Retrieval。Lifecycle、Security、Conversation 和 API 都通过
这里读取同一个服务端事实来源。每个公开写操作都使用 SQLite 事务；SQLite 与 Chroma
之间无法形成跨数据库原子事务，因此跨系统恢复策略由 ``lifecycle.py`` 明确处理。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    """返回带时区的 UTC ISO 时间，便于 API/数据库稳定序列化。"""
    return datetime.now(UTC).isoformat()


class CatalogError(RuntimeError):
    """Catalog 的业务异常基类。"""


class NotFoundError(CatalogError):
    """请求的租户、文档、版本或会话不存在。"""


class ConflictError(CatalogError):
    """状态转换或唯一性约束冲突。"""


class DocumentCatalog:
    """SQLite Catalog 门面。

    参数：
        path: V3 自己的 SQLite 文件。测试传临时路径，避免污染真实 runtime。

    线程模型：
        每次操作创建短生命周期连接，不在线程之间共享 sqlite3.Connection。WAL 允许读写
        更好地并行；真正的生产环境仍应考虑 PostgreSQL、迁移工具和分布式锁。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """提供会在代码块结束时真正 close 的普通连接。

        sqlite3.Connection 自己的 ``with`` 只负责事务提交/回滚，并不会关闭文件句柄；
        Windows 上这会导致临时数据库无法删除，所以这里显式管理生命周期。
        """
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """提供自动 commit/rollback 的事务上下文。"""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """幂等创建 Phase 9 所需表和索引。"""
        schema = """
        CREATE TABLE IF NOT EXISTS tenants (
            tenant_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            tenant_id TEXT NOT NULL,
            knowledge_base_id TEXT NOT NULL,
            name TEXT NOT NULL,
            epoch INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            PRIMARY KEY (tenant_id, knowledge_base_id),
            FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
        );
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            username TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            roles_json TEXT NOT NULL,
            groups_json TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            token_version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            UNIQUE (tenant_id, username),
            FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
        );
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            knowledge_base_id TEXT NOT NULL,
            document_name TEXT NOT NULL,
            source TEXT NOT NULL,
            document_type TEXT NOT NULL,
            classification TEXT NOT NULL,
            owner TEXT NOT NULL,
            status TEXT NOT NULL,
            active_version_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (tenant_id, knowledge_base_id, source),
            FOREIGN KEY (tenant_id, knowledge_base_id)
                REFERENCES knowledge_bases(tenant_id, knowledge_base_id)
        );
        CREATE TABLE IF NOT EXISTS document_versions (
            version_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            version_number INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            storage_path TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (document_id, version_number),
            UNIQUE (document_id, content_hash),
            FOREIGN KEY (document_id) REFERENCES documents(document_id)
        );
        CREATE TABLE IF NOT EXISTS document_acl (
            document_id TEXT NOT NULL,
            subject_type TEXT NOT NULL,
            subject_id TEXT NOT NULL,
            permission TEXT NOT NULL,
            PRIMARY KEY (document_id, subject_type, subject_id, permission),
            FOREIGN KEY (document_id) REFERENCES documents(document_id)
        );
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            knowledge_base_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS ix_documents_scope
            ON documents(tenant_id, knowledge_base_id, status, active_version_id);
        CREATE INDEX IF NOT EXISTS ix_versions_document
            ON document_versions(document_id, version_number);
        CREATE INDEX IF NOT EXISTS ix_messages_conversation
            ON messages(conversation_id, message_id);
        """
        with self.connection() as connection:
            connection.executescript(schema)

    @staticmethod
    def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def ensure_tenant(self, tenant_id: str, name: str | None = None) -> None:
        """幂等创建租户。tenant_id 来自管理操作，不来自普通检索请求。"""
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO tenants VALUES (?, ?, ?)",
                (tenant_id, name or tenant_id, utc_now()),
            )

    def ensure_knowledge_base(
        self, tenant_id: str, knowledge_base_id: str, name: str | None = None
    ) -> None:
        """幂等创建租户内知识库。"""
        self.ensure_tenant(tenant_id)
        with self.transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO knowledge_bases
                   (tenant_id, knowledge_base_id, name, epoch, created_at)
                   VALUES (?, ?, ?, 0, ?)""",
                (tenant_id, knowledge_base_id, name or knowledge_base_id, utc_now()),
            )

    def knowledge_base_epoch(self, tenant_id: str, knowledge_base_id: str) -> int:
        """读取知识库版本 epoch；发布/回滚/删除时递增，用于 Cache 隔离。"""
        with self.connection() as connection:
            row = connection.execute(
                "SELECT epoch FROM knowledge_bases WHERE tenant_id=? AND knowledge_base_id=?",
                (tenant_id, knowledge_base_id),
            ).fetchone()
        if row is None:
            raise NotFoundError("Knowledge base not found")
        return int(row["epoch"])

    @staticmethod
    def _bump_epoch(
        connection: sqlite3.Connection, tenant_id: str, knowledge_base_id: str
    ) -> None:
        connection.execute(
            """UPDATE knowledge_bases SET epoch=epoch+1
               WHERE tenant_id=? AND knowledge_base_id=?""",
            (tenant_id, knowledge_base_id),
        )

    def create_user(
        self,
        *,
        tenant_id: str,
        username: str,
        password_hash: str,
        roles: list[str],
        groups: list[str] | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """创建服务端用户。roles/groups 只在此管理入口写入，不信任聊天请求字段。"""
        self.ensure_tenant(tenant_id)
        identifier = user_id or uuid.uuid4().hex
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO users
                   (user_id, tenant_id, username, password_hash, roles_json,
                    groups_json, active, token_version, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 1, 1, ?)""",
                (
                    identifier,
                    tenant_id,
                    username,
                    password_hash,
                    json.dumps(sorted(set(roles))),
                    json.dumps(sorted(set(groups or []))),
                    utc_now(),
                ),
            )
        return self.get_user(identifier)

    def get_user(self, user_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE user_id=?", (user_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError("User not found")
        value = dict(row)
        value["roles"] = json.loads(value.pop("roles_json"))
        value["groups"] = json.loads(value.pop("groups_json"))
        value["active"] = bool(value["active"])
        return value

    def find_user(self, tenant_id: str, username: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT user_id FROM users WHERE tenant_id=? AND username=?",
                (tenant_id, username),
            ).fetchone()
        return self.get_user(row["user_id"]) if row else None

    def register_document(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        document_name: str,
        source: str,
        document_type: str,
        classification: str,
        owner: str,
        document_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """注册逻辑 Document；同租户/知识库/source 重复注册返回原记录。"""
        self.ensure_knowledge_base(tenant_id, knowledge_base_id)
        now = utc_now()
        identifier = document_id or uuid.uuid4().hex
        with self.transaction() as connection:
            existing = connection.execute(
                """SELECT * FROM documents
                   WHERE tenant_id=? AND knowledge_base_id=? AND source=?""",
                (tenant_id, knowledge_base_id, source),
            ).fetchone()
            if existing:
                return dict(existing), False
            connection.execute(
                """INSERT INTO documents
                   (document_id, tenant_id, knowledge_base_id, document_name,
                    source, document_type, classification, owner, status,
                    active_version_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', NULL, ?, ?)""",
                (
                    identifier,
                    tenant_id,
                    knowledge_base_id,
                    document_name,
                    source,
                    document_type,
                    classification,
                    owner,
                    now,
                    now,
                ),
            )
        return self.get_document(identifier), True

    def get_document(self, document_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id=?", (document_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError("Document not found")
        return dict(row)

    def list_documents(
        self, tenant_id: str, knowledge_base_id: str, *, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM documents WHERE tenant_id=? AND knowledge_base_id=?"
        params: list[Any] = [tenant_id, knowledge_base_id]
        if not include_deleted:
            sql += " AND status!='deleted'"
        sql += " ORDER BY created_at, document_id"
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(sql, params).fetchall()]

    def create_version(
        self,
        *,
        document_id: str,
        content_hash: str,
        storage_path: str,
        version_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """注册不可变版本；相同 document/content_hash 返回已有版本，保证幂等。"""
        identifier = version_id or uuid.uuid4().hex
        now = utc_now()
        with self.transaction() as connection:
            existing = connection.execute(
                """SELECT * FROM document_versions
                   WHERE document_id=? AND content_hash=?""",
                (document_id, content_hash),
            ).fetchone()
            if existing:
                return dict(existing), False
            number = connection.execute(
                """SELECT COALESCE(MAX(version_number), 0) + 1 AS number
                   FROM document_versions WHERE document_id=?""",
                (document_id,),
            ).fetchone()["number"]
            connection.execute(
                """INSERT INTO document_versions
                   (version_id, document_id, version_number, content_hash,
                    storage_path, status, error, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 'registered', NULL, ?, ?)""",
                (identifier, document_id, number, content_hash, storage_path, now, now),
            )
        return self.get_version(identifier), True

    def get_version(self, version_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM document_versions WHERE version_id=?", (version_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError("Document version not found")
        return dict(row)

    def list_versions(self, document_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT * FROM document_versions WHERE document_id=?
                   ORDER BY version_number DESC""",
                (document_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_version_status(
        self, version_id: str, status: str, error: str | None = None
    ) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                """UPDATE document_versions SET status=?, error=?, updated_at=?
                   WHERE version_id=?""",
                (status, error, utc_now(), version_id),
            )
            if cursor.rowcount != 1:
                raise NotFoundError("Document version not found")

    def publish_version(self, document_id: str, version_id: str) -> dict[str, Any]:
        """原子切换 Catalog active_version；Chroma Chunk 已在发布前完成索引。"""
        with self.transaction() as connection:
            document = connection.execute(
                "SELECT * FROM documents WHERE document_id=?", (document_id,)
            ).fetchone()
            version = connection.execute(
                """SELECT * FROM document_versions
                   WHERE version_id=? AND document_id=?""",
                (version_id, document_id),
            ).fetchone()
            if document is None or version is None:
                raise NotFoundError("Document or version not found")
            if document["status"] == "deleted":
                raise ConflictError("Deleted document must be restored before publish")
            if version["status"] not in {"indexed", "published", "retired"}:
                raise ConflictError("Only an indexed version can be published")
            old = document["active_version_id"]
            # 同一版本已发布时是真正的幂等操作：不重复增加 epoch，不制造无意义 Cache Miss。
            if old == version_id and version["status"] == "published":
                return dict(document)
            if old and old != version_id:
                connection.execute(
                    """UPDATE document_versions SET status='retired', updated_at=?
                       WHERE version_id=?""",
                    (utc_now(), old),
                )
            connection.execute(
                """UPDATE document_versions SET status='published', updated_at=?
                   WHERE version_id=?""",
                (utc_now(), version_id),
            )
            connection.execute(
                """UPDATE documents SET active_version_id=?, status='active', updated_at=?
                   WHERE document_id=?""",
                (version_id, utc_now(), document_id),
            )
            self._bump_epoch(
                connection, document["tenant_id"], document["knowledge_base_id"]
            )
        return self.get_document(document_id)

    def soft_delete_document(self, document_id: str) -> None:
        document = self.get_document(document_id)
        with self.transaction() as connection:
            connection.execute(
                "UPDATE documents SET status='deleted', updated_at=? WHERE document_id=?",
                (utc_now(), document_id),
            )
            self._bump_epoch(
                connection, document["tenant_id"], document["knowledge_base_id"]
            )

    def restore_document(self, document_id: str) -> None:
        document = self.get_document(document_id)
        with self.transaction() as connection:
            connection.execute(
                "UPDATE documents SET status='active', updated_at=? WHERE document_id=?",
                (utc_now(), document_id),
            )
            self._bump_epoch(
                connection, document["tenant_id"], document["knowledge_base_id"]
            )

    def grant_acl(
        self, document_id: str, subject_type: str, subject_id: str, permission: str = "read"
    ) -> None:
        """授予文档 ACL。subject_type 只能是 user/role/group。"""
        if subject_type not in {"user", "role", "group"}:
            raise ValueError("Invalid ACL subject type")
        with self.transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO document_acl
                   (document_id, subject_type, subject_id, permission)
                   VALUES (?, ?, ?, ?)""",
                (document_id, subject_type, subject_id, permission),
            )

    def active_documents(self, tenant_id: str, knowledge_base_id: str) -> list[dict[str, Any]]:
        """返回当前 published 且未删除的逻辑文档和版本。"""
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT d.*, v.version_number, v.storage_path, v.content_hash,
                          v.status AS version_status
                   FROM documents d
                   JOIN document_versions v ON v.version_id=d.active_version_id
                   WHERE d.tenant_id=? AND d.knowledge_base_id=?
                     AND d.status='active' AND v.status='published'""",
                (tenant_id, knowledge_base_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def acl_entries(self, document_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM document_acl WHERE document_id=?", (document_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def create_conversation(
        self, tenant_id: str, user_id: str, knowledge_base_id: str
    ) -> dict[str, Any]:
        identifier = uuid.uuid4().hex
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?)""",
                (identifier, tenant_id, user_id, knowledge_base_id, now, now),
            )
        return self.get_conversation(identifier, tenant_id, user_id)

    def get_conversation(
        self, conversation_id: str, tenant_id: str, user_id: str
    ) -> dict[str, Any]:
        """按 tenant+owner 读取会话；不匹配时统一返回 Not Found，避免枚举。"""
        with self.connection() as connection:
            row = connection.execute(
                """SELECT * FROM conversations
                   WHERE conversation_id=? AND tenant_id=? AND user_id=?""",
                (conversation_id, tenant_id, user_id),
            ).fetchone()
        if row is None:
            raise NotFoundError("Conversation not found")
        return dict(row)

    def add_message(self, conversation_id: str, role: str, content: str) -> dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError("Conversation role must be user or assistant")
        now = utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                """INSERT INTO messages(conversation_id, role, content, created_at)
                   VALUES (?, ?, ?, ?)""",
                (conversation_id, role, content, now),
            )
            connection.execute(
                "UPDATE conversations SET updated_at=? WHERE conversation_id=?",
                (now, conversation_id),
            )
            message_id = cursor.lastrowid
        return {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "created_at": now,
        }

    def list_messages(self, conversation_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """按时间顺序返回最近 limit 条消息。"""
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT * FROM (
                       SELECT * FROM messages WHERE conversation_id=?
                       ORDER BY message_id DESC LIMIT ?
                   ) ORDER BY message_id""",
                (conversation_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_conversation(
        self, conversation_id: str, tenant_id: str, user_id: str
    ) -> None:
        self.get_conversation(conversation_id, tenant_id, user_id)
        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM conversations WHERE conversation_id=?",
                (conversation_id,),
            )

