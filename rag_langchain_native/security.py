"""Phase 9 身份认证、RBAC/ACL 授权与 Authorized Retrieval Scope。

Authentication（认证）回答“你是谁”：密码登录后签发 HMAC 签名 Token，后续请求必须验证
签名、过期时间、用户状态和 token_version。Authorization（授权）回答“你能访问什么”：
服务端根据数据库中的 tenant/roles/groups 与 Document ACL 计算可见版本。

客户端不得通过 JSON 自行声明 tenant_id 或 role。Vector 与 BM25 都接收同一个
``AuthorizationScope``，权限检查发生在候选检索阶段，未授权 Chunk 不会进入 Fusion、
Reranker、Context 或 Citation。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from .catalog import DocumentCatalog, NotFoundError


class AuthenticationError(RuntimeError):
    """Token、密码或服务端认证配置无效。"""


class AuthorizationError(RuntimeError):
    """身份有效，但没有执行目标操作的权限。"""


@dataclass(frozen=True)
class Principal:
    """已由服务端验证的调用者身份，不接受客户端直接构造为信任依据。"""

    user_id: str
    tenant_id: str
    roles: tuple[str, ...]
    groups: tuple[str, ...]

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles


@dataclass(frozen=True)
class AuthorizationScope:
    """一次检索的不可变授权范围。

    ``version_ids`` 来自当前 active/published 且 Principal 有权读取的文档。Chroma 使用
    ``metadata_filter``，内存 BM25 则使用同一批 version_ids 过滤，避免两条路线越权。
    """

    tenant_id: str
    knowledge_base_id: str
    document_ids: tuple[str, ...]
    version_ids: tuple[str, ...]
    epoch: int

    @property
    def empty(self) -> bool:
        return not self.version_ids

    def metadata_filter(self) -> dict[str, Any] | None:
        if self.empty:
            return None
        return {
            "$and": [
                {"tenant_id": {"$eq": self.tenant_id}},
                {"knowledge_base_id": {"$eq": self.knowledge_base_id}},
                {"version_id": {"$in": list(self.version_ids)}},
            ]
        }

    @property
    def cache_namespace(self) -> str:
        versions = ",".join(self.version_ids)
        digest = hashlib.sha256(versions.encode("utf-8")).hexdigest()[:16]
        return f"{self.tenant_id}:{self.knowledge_base_id}:{self.epoch}:{digest}"


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """使用 PBKDF2-HMAC-SHA256 保存密码，不存储明文。"""
    if len(password) < 8:
        raise ValueError("Password must contain at least 8 characters")
    actual_salt = salt or os.urandom(16)
    iterations = 260_000
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), actual_salt, iterations
    )
    return "pbkdf2_sha256${}${}${}".format(
        iterations, actual_salt.hex(), digest.hex()
    )


def verify_password(password: str, encoded: str) -> bool:
    """常量时间比较 PBKDF2 密码摘要。格式损坏时返回 False。"""
    try:
        algorithm, iterations, salt_hex, expected_hex = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
        return hmac.compare_digest(actual.hex(), expected_hex)
    except (ValueError, TypeError):
        return False


def _b64_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    decoded = base64.urlsafe_b64decode(value + padding)
    # Base64 最后几个未使用 bit 可能产生多个文本表示。只接受规范编码，避免同一签名
    # 出现不同 Token 字符串，便于撤销、审计和精确比较。
    if _b64_encode(decoded) != value:
        raise ValueError("Non-canonical base64 encoding")
    return decoded


class AuthService:
    """创建并验证本地学习环境的签名 Token。

    这不是 OAuth/OIDC 的替代品。它用于展示“服务端验证身份”的最小闭环；生产环境应
    接入企业 IdP、密钥轮换、撤销列表、MFA、审计和 HTTPS。
    """

    def __init__(
        self,
        catalog: DocumentCatalog,
        secret: str | None,
        token_ttl_seconds: int = 3600,
        issuer: str = "rag-langchain-native",
        audience: str = "rag-api",
    ) -> None:
        self.catalog = catalog
        self._secret = secret.encode("utf-8") if secret else None
        self.token_ttl_seconds = token_ttl_seconds
        self.issuer = issuer
        self.audience = audience

    def _require_secret(self) -> bytes:
        if not self._secret:
            raise AuthenticationError("V3_AUTH_SECRET is not configured")
        return self._secret

    def register_user(
        self,
        *,
        tenant_id: str,
        username: str,
        password: str,
        roles: list[str],
        groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """管理/CLI 入口：创建用户并保存密码摘要。"""
        invalid = set(roles) - {"general", "hr", "finance", "admin"}
        if invalid:
            raise ValueError(f"Unsupported roles: {sorted(invalid)}")
        return self.catalog.create_user(
            tenant_id=tenant_id,
            username=username,
            password_hash=hash_password(password),
            roles=roles or ["general"],
            groups=groups,
        )

    def login(self, tenant_id: str, username: str, password: str) -> str:
        """验证密码并签发只携带 user_id/token_version 的 Token。"""
        user = self.catalog.find_user(tenant_id, username)
        if (
            not user
            or not user["active"]
            or not verify_password(password, user["password_hash"])
        ):
            raise AuthenticationError("Invalid credentials")
        now = int(time.time())
        payload = {
            "sub": user["user_id"],
            "tv": int(user["token_version"]),
            "iat": now,
            "exp": now + self.token_ttl_seconds,
            "iss": self.issuer,
            "aud": self.audience,
        }
        body = _b64_encode(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        signature = hmac.new(
            self._require_secret(), body.encode("ascii"), hashlib.sha256
        ).digest()
        return f"{body}.{_b64_encode(signature)}"

    def verify_token(self, token: str) -> Principal:
        """验证签名并重新从 Catalog 加载 tenant/roles/groups。

        即使攻击者修改 Token 内容，也无法生成正确签名；即使旧 Token 签名正确，禁用用户
        或提高 token_version 后也会失败。最关键的是权限字段不直接信任 Token Payload。
        """
        try:
            body, signature_text = token.split(".", 1)
            expected = hmac.new(
                self._require_secret(), body.encode("ascii"), hashlib.sha256
            ).digest()
            if not hmac.compare_digest(expected, _b64_decode(signature_text)):
                raise AuthenticationError("Invalid token signature")
            payload = json.loads(_b64_decode(body))
            if payload.get("iss") != self.issuer or payload.get("aud") != self.audience:
                raise AuthenticationError("Token issuer or audience mismatch")
            if int(payload["exp"]) < int(time.time()):
                raise AuthenticationError("Token expired")
            user = self.catalog.get_user(str(payload["sub"]))
            if not user["active"] or int(payload["tv"]) != int(user["token_version"]):
                raise AuthenticationError("Token revoked")
        except AuthenticationError:
            raise
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, NotFoundError) as error:
            raise AuthenticationError("Invalid token") from error
        return Principal(
            user_id=user["user_id"],
            tenant_id=user["tenant_id"],
            roles=tuple(user["roles"]),
            groups=tuple(user["groups"]),
        )


def can_read_document(
    catalog: DocumentCatalog, principal: Principal, document: dict[str, Any]
) -> bool:
    """同时执行 tenant、classification RBAC 和显式 ACL 判断。"""
    if document["tenant_id"] != principal.tenant_id:
        return False
    if principal.is_admin:
        return True
    classification = str(document["classification"]).lower()
    role_allowed = classification == "general" or classification in principal.roles
    if role_allowed:
        return True
    for entry in catalog.acl_entries(document["document_id"]):
        if entry["permission"] != "read":
            continue
        if entry["subject_type"] == "user" and entry["subject_id"] == principal.user_id:
            return True
        if entry["subject_type"] == "role" and entry["subject_id"] in principal.roles:
            return True
        if entry["subject_type"] == "group" and entry["subject_id"] in principal.groups:
            return True
    return False


def build_authorization_scope(
    catalog: DocumentCatalog,
    principal: Principal,
    knowledge_base_id: str,
    *,
    document_id: str | None = None,
    version_id: str | None = None,
    classification: str | None = None,
) -> AuthorizationScope:
    """从 Catalog 计算授权 active version，再与客户端可选筛选条件取交集。

    客户端 Filter 只能缩小服务端已授权集合，绝不能扩大它；指定历史 version_id 也不会
    绕过 active/published 约束。
    """
    active = catalog.active_documents(principal.tenant_id, knowledge_base_id)
    allowed = [doc for doc in active if can_read_document(catalog, principal, doc)]
    if document_id is not None:
        allowed = [doc for doc in allowed if doc["document_id"] == document_id]
    if version_id is not None:
        allowed = [doc for doc in allowed if doc["active_version_id"] == version_id]
    if classification is not None:
        allowed = [
            doc
            for doc in allowed
            if str(doc["classification"]).lower() == classification.lower()
        ]
    return AuthorizationScope(
        tenant_id=principal.tenant_id,
        knowledge_base_id=knowledge_base_id,
        document_ids=tuple(sorted(doc["document_id"] for doc in allowed)),
        version_ids=tuple(sorted(doc["active_version_id"] for doc in allowed)),
        epoch=catalog.knowledge_base_epoch(principal.tenant_id, knowledge_base_id),
    )


def require_admin(principal: Principal) -> None:
    """文档上传、发布、删除、回滚等管理操作的统一 fail-closed 检查。"""
    if not principal.is_admin:
        raise AuthorizationError("Administrator role required")


def require_document_access(
    catalog: DocumentCatalog, principal: Principal, document_id: str
) -> dict[str, Any]:
    """验证文档属于当前租户且 Principal 有读取权限，供下载等非检索接口复用。"""
    document = catalog.get_document(document_id)
    if document["status"] != "active":
        raise AuthorizationError("Document is not active")
    if not can_read_document(catalog, principal, document):
        raise AuthorizationError("Document access denied")
    return document

