import os
import shutil

from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi import HTTPException
from fastapi import File
from fastapi import UploadFile

from rag_service import ask_rag
from knowledge_service import add_pdf_to_knowledge, delete_file_from_db

UPLOAD_DIR = "./data/upload"
FILE_MAX_SIZE = 20 * 1024 * 1024

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="RAG DEMO",
    description="A simple RAG demo using FastAPI",
    version="1.0.0",
)

class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1, 
        max_length=1000, 
        example="What is the company's mission statement?"
    )

class sourceItem(BaseModel):
    source: str
    chunk_id: str | int | None = None
    page: int | None = None
    distance: float
    document: str

class QuestionResponse(BaseModel):
    question: str
    answer: str
    source: list[sourceItem]

class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    chunks: int
    status: str

@app.get("/")
def root():
    return health_check()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=QuestionResponse)
def chat(question: QuestionRequest):

    try:
        result = ask_rag(question.question)
    
        source = []

        for metadata, distance, doc in zip(result["metadatas"], result["distances"], result["documents"]):
            source.append(
                {
                    "source": metadata["source"],
                    "chunk_id": metadata["chunk_id"],
                    "page": metadata["page"] if "page" in metadata else "N/A",
                    "distance": distance,
                    "document": doc
                }
            )

        return {
            "question": question.question,
            "answer": result["answer"],
            "source": source
        }
    
    except Exception as e:

        print(
            f"RAG Error: {str(e)}"
        )
            
        raise HTTPException(
            status_code=500, 
            detail="RAG service error. Please try again later."
        ) from e

@app.post(
    "/documents",
    response_model=DocumentUploadResponse
)
def upload_document(
    file: UploadFile = File(...)
):
    try:
        # 1. 检查文件名
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Missing filename"
            )


        # 2. 只允许pdf
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF file are supported"
            )

        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400,
                detail="Only PDF file are supported"
            )
        
        # 3.生成本地路径
        file_path = os.path.join(
            UPLOAD_DIR,
            file.filename
        )

        # 4. 保存上传文件
        with open(
            file_path,
            "wb"
        ) as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        file_size = os.path.getsize(UPLOAD_DIR)

        if file_size > FILE_MAX_SIZE:
            os.remove(UPLOAD_DIR)

            raise HTTPException(
                status_code=400,
                detail="The size of file should smaller than 20MB  "
            ) 

        # 5.自动入库
        result = add_pdf_to_knowledge(file_path)
        

        return {
            "message": "Document uploaded and indexed",
            "filename": result["filename"],
            "chunks": result["chunks"],
            "status": result["status"]
        }
    
    except HTTPException:
        raise

    except Exception as e:
        print(
            f"Document Upload Error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Document indexing failed"
        ) from e

@app.get("/documents")
def list_documents():

    files = []

    for filename in os.listdir(UPLOAD_DIR):
        if filename.endswith(".pdf"):
            files.append(filename)

    return {
        "count": len(files),
        "file": files
    }

@app.delete("/documents/{filename}")
def delete_file(
    filename: str
):
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="File not exists!"
        )

    delete_file_from_db(filename)

    os.remove(file_path)

    return{
        "message": "Document deleted",
        "filename": filename
    }
    