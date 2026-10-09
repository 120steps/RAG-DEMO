"""LangChain V3 的独立 FastAPI 入口（约定端口 8011）。

文件职责：
    把 V3 的健康检查、Retrieval-only、完整问答和 PDF Upload 暴露为 HTTP API。

在 RAG Pipeline 中的位置：
    API 是最上游适配层。它负责把 HTTP JSON/文件转换成 Python 参数，然后调用
    ``NativeRAGService`` 或 Ingestion；它不重新实现 Retrieval、Prompt 或 Citation。

输入与输出：
    ``/retrieve`` 和 ``/chat`` 接收 Pydantic ``QueryRequest``；``/upload`` 接收 PDF。
    下游返回的普通 dict 会由 FastAPI 自动序列化为 JSON。

隔离原则：
    本文件不修改根 ``app.py``，不调用 V1/V2 服务。上传文件、Chroma 和 Cache 全部写入
    V3 runtime。当前 ``/chat`` 是同步完整响应，不是 SSE token streaming。

LangChain 关系：
    FastAPI 本身不是 LangChain。它通过 ``get_service().ask_rag()`` 进入 LCEL Chain。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .chain import get_service
from .config import DEFAULT_SETTINGS, Settings
from .ingestion import ingest_pdf


class QueryRequest(BaseModel):
    """Retrieval/Chat 的 HTTP 请求模型。

    ``question`` 至少一个字符；``top_k`` 只用于 Retrieval-only 路由，并限制在 1 到 50，
    防止客户端请求无意义或过大的候选数量。
    """

    question: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)


def create_app(settings: Settings = DEFAULT_SETTINGS) -> FastAPI:
    """创建并配置独立 V3 FastAPI 应用。

    参数：
        settings (Settings): V3 配置；测试可注入临时 runtime 配置。

    返回：
        FastAPI: 已注册 health、retrieve、chat、upload 路由的应用对象。

    调用关系：
        模块末尾 ``app = create_app()`` 供 Uvicorn 导入；各路由再调用共享 Service。

    初学者知识点：
        路由装饰器把内部函数注册到 HTTP 路径。闭包使这些函数可以读取外层 ``settings``，
        不需要把配置暴露为全局可变变量。
    """
    application = FastAPI(
        title="LangChain Native RAG V3",
        version="0.1.0",
    )

    @application.get("/health")
    def health() -> dict:
        """返回进程状态和 V3 数据隔离信息，不加载/调用 Gemini。"""
        return {
            "status": "ok",
            "version": "v3",
            "collection": settings.collection_name,
            "runtime": str(settings.runtime_dir),
        }

    @application.post("/retrieve")
    def retrieve(request: QueryRequest) -> dict:
        """执行 Retrieval-only 调试，不调用最终 Answer Generation。

        输入 Pydantic 对象，输出 Query、Documents、分数和延迟字典。异常被转换为 HTTP
        500，并保留原异常作为 ``from error`` 的 cause，方便服务日志排查。
        """
        try:
            return get_service().retrieve_only(
                request.question, top_k=request.top_k
            )
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @application.post("/chat")
    def chat(request: QueryRequest) -> dict:
        """同步执行 V3 完整 RAG Chain，并返回 Answer、Citation 与 Debug 字段。

        当前函数是普通 ``def``，调用 ``ask_rag()->invoke()``，会等待完整回答。虽然
        Service 提供 ``aask_rag``，这里尚未使用 ``async def/await``，也没有 SSE。
        """
        try:
            return get_service().ask_rag(request.question)
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @application.post("/upload")
    async def upload(file: UploadFile = File(...)) -> dict:
        """接收一个 PDF，保存到 V3 runtime 并写入同一知识库。

        参数：
            file (UploadFile): FastAPI 对 multipart 上传文件的封装。

        返回：
            dict: Ingestion 的 source、页数、Chunk 数和 IDs。

        执行过程：
            1. ``Path(...).name`` 去掉客户端可能携带的目录，只保留安全文件名。
            2. 检查扩展名，创建 V3 runtime 目录。
            3. ``await file.read()`` 异步读取上传内容，再写入 V3 uploads。
            4. 复用 Service 的 VectorStore 调 ``ingest_pdf``。
            5. ``refresh_retriever`` 重建内存 BM25，并让下次请求重建 Chain。

        初学者知识点：
            ``async def`` 定义异步路由；``await`` 暂停当前协程等待文件读取。后续 PDF
            解析和 Embedding 仍是同步计算，并不会因为路由是 async 自动变成非阻塞。
        """
        filename = Path(file.filename or "").name
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        settings.ensure_runtime_dirs()
        destination = settings.upload_dir / filename
        # ``await`` 只能出现在 async def 中；读取完成后 content 的类型是 bytes。
        content = await file.read()
        destination.write_bytes(content)
        try:
            result = ingest_pdf(
                destination,
                settings=settings,
                vectorstore=get_service().vectorstore,
            )
            get_service().refresh_retriever()
            return result
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    return application


# Uvicorn 使用 ``rag_langchain_native.api:app`` 导入这个模块级应用对象。
app = create_app()

