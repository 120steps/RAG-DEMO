from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi import HTTPException
from rag_service import ask_rag

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