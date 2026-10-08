"""Independent FastAPI application for V3 (default port: 8011)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .chain import get_service
from .config import DEFAULT_SETTINGS, Settings
from .ingestion import ingest_pdf


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)


def create_app(settings: Settings = DEFAULT_SETTINGS) -> FastAPI:
    application = FastAPI(
        title="LangChain Native RAG V3",
        version="0.1.0",
    )

    @application.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": "v3",
            "collection": settings.collection_name,
            "runtime": str(settings.runtime_dir),
        }

    @application.post("/retrieve")
    def retrieve(request: QueryRequest) -> dict:
        try:
            return get_service().retrieve_only(
                request.question, top_k=request.top_k
            )
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @application.post("/chat")
    def chat(request: QueryRequest) -> dict:
        try:
            return get_service().ask_rag(request.question)
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @application.post("/upload")
    async def upload(file: UploadFile = File(...)) -> dict:
        filename = Path(file.filename or "").name
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        settings.ensure_runtime_dirs()
        destination = settings.upload_dir / filename
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


app = create_app()

