from pydantic import BaseModel, Field
from typing import Literal

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunks: int
    status: Literal["indexed"] = "indexed"

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[str] = Field(default_factory=list, description="Empty means search all indexed documents")

class Citation(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    page: int | None = None
    score: float
    text: str

class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    rewritten_question: str
    grounded: bool
    confidence: Literal["low", "medium", "high"]
    workflow: list[str]

class HealthResponse(BaseModel):
    status: str
    model: str
    indexed_documents: int
