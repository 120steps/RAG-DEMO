"""Phase 9 企业 FastAPI：认证、会话、授权问答和文档生命周期的统一 HTTP 入口。

除 ``/health`` 与 ``/auth/login`` 外，所有路由都依赖 HTTP Bearer Token。Token 只用于定位
用户，tenant/roles/groups 每次从服务端 Catalog 重新加载；请求体没有可被信任的 role 或
tenant 字段。管理操作要求 admin，下载和检索使用同一 ACL/RBAC 判断。

应用仍运行在 V3 独立端口 8011，不修改根目录 FastAPI。重型 Embedding/Chroma 对象按需
创建，因此 ``/health`` 不会下载模型或调用 Gemini。
"""

from __future__ import annotations

from pathlib import Path
import asyncio

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Security,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from .catalog import CatalogError, DocumentCatalog, NotFoundError
from .config import DEFAULT_SETTINGS, Settings
from .enterprise import EnterpriseRAGService
from .lifecycle import DocumentLifecycleService, validate_pdf_content
from .rate_limit import LocalRateLimiter
from .security import (
    AuthService,
    AuthenticationError,
    AuthorizationError,
    Principal,
    can_read_document,
    require_admin,
    require_document_access,
)


class LoginRequest(BaseModel):
    tenant_id: str = Field(min_length=1)
    username: str = Field(min_length=1)
    password: str = Field(min_length=8)


class QueryRequest(BaseModel):
    """客户端只选择知识库与会话，不能声明 tenant 或 role。"""

    question: str = Field(min_length=1, max_length=4000)
    knowledge_base_id: str = Field(default="default", min_length=1)
    conversation_id: str | None = None
    document_id: str | None = None
    version_id: str | None = None
    classification: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)


class ConversationRequest(BaseModel):
    knowledge_base_id: str = Field(default="default", min_length=1)


class RegisterDocumentRequest(BaseModel):
    knowledge_base_id: str = Field(default="default", min_length=1)
    document_name: str = Field(min_length=1)
    source: str = Field(min_length=1)
    document_type: str = "pdf"
    classification: str = "general"


class VersionActionRequest(BaseModel):
    version_id: str = Field(min_length=1)


def create_app(
    settings: Settings = DEFAULT_SETTINGS,
    *,
    catalog: DocumentCatalog | None = None,
    vectorstore=None,
    llm: Runnable | None = None,
    query_model: Runnable | None = None,
    reranker=None,
) -> FastAPI:
    """创建 Phase 9 API；测试可注入临时 Catalog、Chroma 和 Fake LLM。"""
    settings.validate_security()
    application = FastAPI(
        title="LangChain Native Enterprise RAG V3",
        version="0.4.0",
        docs_url=None if settings.environment == "production" else "/docs",
        redoc_url=None if settings.environment == "production" else "/redoc",
    )
    if settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        )
    document_catalog = catalog or DocumentCatalog(settings.catalog_path)
    auth = AuthService(
        document_catalog,
        settings.auth_secret,
        settings.auth_token_ttl_seconds,
        settings.auth_issuer,
        settings.auth_audience,
    )
    bearer = HTTPBearer(auto_error=False)
    lifecycle_instance: DocumentLifecycleService | None = None
    enterprise_instance: EnterpriseRAGService | None = None
    limiter = LocalRateLimiter(settings.api_rate_limit_per_minute)
    concurrency = asyncio.Semaphore(settings.max_concurrent_requests)

    def secure_response(response):
        """统一补充浏览器安全头；包括中间件提前拒绝的 413/429 响应。"""
        response.headers.update(
            {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "Referrer-Policy": "no-referrer",
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
            }
        )
        return response

    @application.middleware("http")
    async def security_middleware(request: Request, call_next):
        """执行请求大小、单进程限流、并发背压和安全响应头。"""
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > settings.max_request_bytes):
            return secure_response(
                JSONResponse(status_code=413, content={"detail": "Request body too large"})
            )
        client = request.client.host if request.client else "unknown"
        allowed, retry_after = limiter.allow(client)
        if not allowed:
            return secure_response(
                JSONResponse(
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                    content={"detail": "Rate limit exceeded"},
                )
            )
        async with concurrency:
            response = await call_next(request)
        return secure_response(response)

    async def read_upload(file: UploadFile) -> bytes:
        """分块读取并在超过预算时立即拒绝，避免一次无限制读入内存。"""
        chunks: list[bytes] = []
        total = 0
        while chunk := await file.read(min(1024 * 1024, settings.max_upload_bytes + 1)):
            total += len(chunk)
            if total > settings.max_upload_bytes:
                raise HTTPException(status_code=413, detail="PDF exceeds upload size limit")
            chunks.append(chunk)
        return b"".join(chunks)

    def validate_question(question: str) -> None:
        """执行可配置字符预算；Pydantic 的静态上限之外还允许生产环境收紧。"""
        if len(question) > settings.max_query_chars:
            raise HTTPException(status_code=422, detail="Question exceeds configured limit")

    def lifecycle() -> DocumentLifecycleService:
        nonlocal lifecycle_instance
        if lifecycle_instance is None:
            lifecycle_instance = DocumentLifecycleService(
                document_catalog, settings, vectorstore=vectorstore
            )
        return lifecycle_instance

    def enterprise() -> EnterpriseRAGService:
        nonlocal enterprise_instance
        if enterprise_instance is None:
            enterprise_instance = EnterpriseRAGService(
                document_catalog,
                settings,
                vectorstore=vectorstore,
                llm=llm,
                query_model=query_model,
                reranker=reranker,
            )
        return enterprise_instance

    def current_principal(
        credentials: HTTPAuthorizationCredentials | None = Security(bearer),
    ) -> Principal:
        """FastAPI Security Dependency：验证 Bearer Token，失败默认拒绝。"""
        if credentials is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        try:
            return auth.verify_token(credentials.credentials)
        except AuthenticationError as error:
            raise HTTPException(status_code=401, detail=str(error)) from error

    def checked_admin(principal: Principal = Depends(current_principal)) -> Principal:
        try:
            require_admin(principal)
            return principal
        except AuthorizationError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error

    def ensure_owned_document(principal: Principal, document_id: str) -> dict:
        document = document_catalog.get_document(document_id)
        if document["tenant_id"] != principal.tenant_id:
            raise HTTPException(status_code=404, detail="Document not found")
        return document

    @application.exception_handler(CatalogError)
    async def catalog_error_handler(_, error: CatalogError):
        status = 404 if isinstance(error, NotFoundError) else 409
        return JSONResponse(status_code=status, content={"detail": str(error)})

    @application.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": "v3-phase12",
            "collection": settings.enterprise_collection_name,
            "auth_configured": bool(settings.auth_secret),
        }

    @application.get("/ready")
    def readiness() -> dict:
        """只检查本地关键依赖，不调用 Gemini 或下载模型。"""
        try:
            with document_catalog.connection() as connection:
                connection.execute("SELECT 1").fetchone()
            settings.runtime_dir.mkdir(parents=True, exist_ok=True)
            return {"status": "ready", "catalog": "ok", "runtime": "ok"}
        except Exception:
            return JSONResponse(status_code=503, content={"status": "not_ready"})

    @application.post("/auth/login")
    def login(request: LoginRequest) -> dict:
        try:
            token = auth.login(
                request.tenant_id, request.username, request.password
            )
            return {"access_token": token, "token_type": "bearer"}
        except AuthenticationError as error:
            raise HTTPException(status_code=401, detail=str(error)) from error

    @application.post("/conversations")
    def create_conversation(
        request: ConversationRequest,
        principal: Principal = Depends(current_principal),
    ) -> dict:
        return enterprise().conversations.create(
            principal, request.knowledge_base_id
        )

    @application.get("/conversations/{conversation_id}")
    def get_conversation(
        conversation_id: str,
        principal: Principal = Depends(current_principal),
    ) -> dict:
        return enterprise().conversations.get(principal, conversation_id)

    @application.delete("/conversations/{conversation_id}")
    def delete_conversation(
        conversation_id: str,
        principal: Principal = Depends(current_principal),
    ) -> dict:
        enterprise().conversations.delete(principal, conversation_id)
        return {"deleted": True, "conversation_id": conversation_id}

    @application.post("/retrieve")
    def retrieve(
        request: QueryRequest,
        principal: Principal = Depends(current_principal),
    ) -> dict:
        validate_question(request.question)
        try:
            return enterprise().retrieve_only(
                principal=principal,
                knowledge_base_id=request.knowledge_base_id,
                question=request.question,
                top_k=request.top_k,
                conversation_id=request.conversation_id,
                document_id=request.document_id,
                version_id=request.version_id,
                classification=request.classification,
            )
        except AuthorizationError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error

    @application.post("/chat")
    def chat(
        request: QueryRequest,
        principal: Principal = Depends(current_principal),
    ) -> dict:
        validate_question(request.question)
        try:
            return enterprise().chat(
                principal=principal,
                knowledge_base_id=request.knowledge_base_id,
                question=request.question,
                conversation_id=request.conversation_id,
                document_id=request.document_id,
                version_id=request.version_id,
                classification=request.classification,
            )
        except AuthorizationError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error

    @application.post("/documents/register")
    def register_document(
        request: RegisterDocumentRequest,
        principal: Principal = Depends(checked_admin),
    ) -> dict:
        return lifecycle().register_document(
            tenant_id=principal.tenant_id,
            knowledge_base_id=request.knowledge_base_id,
            document_name=request.document_name,
            source=request.source,
            document_type=request.document_type,
            classification=request.classification,
            owner=principal.user_id,
        )

    @application.post("/documents/{document_id}/versions")
    async def upload_version(
        document_id: str,
        file: UploadFile = File(...),
        index: bool = Form(default=True),
        principal: Principal = Depends(checked_admin),
    ) -> dict:
        document = ensure_owned_document(principal, document_id)
        result = lifecycle().upload_version(
            tenant_id=principal.tenant_id,
            knowledge_base_id=document["knowledge_base_id"],
            document_id=document_id,
            filename=Path(file.filename or document["source"]).name,
            content=await read_upload(file),
        )
        if index and result["status"] in {"registered", "failed"}:
            result = lifecycle().index_version(document_id, result["version_id"])
        return result

    @application.post("/upload")
    async def upload_document(
        file: UploadFile = File(...),
        knowledge_base_id: str = Form(default="default"),
        classification: str = Form(default="general"),
        document_name: str | None = Form(default=None),
        principal: Principal = Depends(checked_admin),
    ) -> dict:
        """便捷入口：注册+上传+索引；发布仍需显式调用，避免未验证版本自动上线。"""
        filename = Path(file.filename or "").name
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        # 先完整完成字节级验证，再注册逻辑文档；恶意/损坏文件不会留下空 Catalog 记录。
        content = await read_upload(file)
        try:
            validate_pdf_content(content, settings)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        document = lifecycle().register_document(
            tenant_id=principal.tenant_id,
            knowledge_base_id=knowledge_base_id,
            document_name=document_name or filename,
            source=filename,
            classification=classification,
            owner=principal.user_id,
        )
        version = lifecycle().upload_version(
            tenant_id=principal.tenant_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document["document_id"],
            filename=filename,
            content=content,
        )
        if version["status"] in {"registered", "failed"}:
            version = lifecycle().index_version(
                document["document_id"], version["version_id"]
            )
        return {"document": document, "version": version}

    @application.get("/documents")
    def list_documents(
        knowledge_base_id: str = Query(default="default"),
        include_deleted: bool = Query(default=False),
        principal: Principal = Depends(current_principal),
    ) -> list[dict]:
        if include_deleted and not principal.is_admin:
            raise HTTPException(status_code=403, detail="Administrator role required")
        documents = lifecycle().list_documents(
            principal.tenant_id,
            knowledge_base_id,
            include_deleted=include_deleted,
        )
        return [
            document
            for document in documents
            if can_read_document(document_catalog, principal, document)
        ]

    @application.get("/documents/{document_id}/versions")
    def list_versions(
        document_id: str,
        principal: Principal = Depends(current_principal),
    ) -> list[dict]:
        try:
            require_document_access(document_catalog, principal, document_id)
        except AuthorizationError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        return lifecycle().list_versions(document_id)

    @application.post("/documents/{document_id}/publish")
    def publish_version(
        document_id: str,
        request: VersionActionRequest,
        principal: Principal = Depends(checked_admin),
    ) -> dict:
        ensure_owned_document(principal, document_id)
        result = lifecycle().publish_version(document_id, request.version_id)
        if enterprise_instance:
            enterprise_instance.invalidate_services()
        return result

    @application.post("/documents/{document_id}/rollback")
    def rollback_version(
        document_id: str,
        request: VersionActionRequest,
        principal: Principal = Depends(checked_admin),
    ) -> dict:
        ensure_owned_document(principal, document_id)
        result = lifecycle().rollback_version(document_id, request.version_id)
        if enterprise_instance:
            enterprise_instance.invalidate_services()
        return result

    @application.delete("/documents/{document_id}")
    def delete_document(
        document_id: str,
        principal: Principal = Depends(checked_admin),
    ) -> dict:
        ensure_owned_document(principal, document_id)
        lifecycle().soft_delete(document_id)
        if enterprise_instance:
            enterprise_instance.invalidate_services()
        return {"deleted": True, "document_id": document_id}

    @application.post("/documents/{document_id}/restore")
    def restore_document(
        document_id: str,
        principal: Principal = Depends(checked_admin),
    ) -> dict:
        ensure_owned_document(principal, document_id)
        lifecycle().restore(document_id)
        if enterprise_instance:
            enterprise_instance.invalidate_services()
        return {"restored": True, "document_id": document_id}

    @application.get("/documents/{document_id}/download")
    def download_document(
        document_id: str,
        version_id: str | None = Query(default=None),
        principal: Principal = Depends(current_principal),
    ):
        try:
            document = require_document_access(
                document_catalog, principal, document_id
            )
        except AuthorizationError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        path = lifecycle().download_path(document_id, version_id)
        return FileResponse(
            path, filename=document["source"], media_type="application/pdf"
        )

    application.state.catalog = document_catalog
    application.state.auth_service = auth
    application.state.get_lifecycle = lifecycle
    application.state.get_enterprise = enterprise
    return application


app = create_app()

