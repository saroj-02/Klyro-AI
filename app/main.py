from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.api.routes import router

settings = get_settings()
app = FastAPI(title=settings.app_name, version="1.0.0", description="Grounded document Q&A with LangGraph, FAISS and Ollama")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501"], allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(router, prefix="/api/v1")

@app.get("/")
def root():
    return {"name": settings.app_name, "docs": "/docs", "health": "/api/v1/health"}
