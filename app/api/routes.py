import hashlib
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.config import get_settings
from app.models import AskRequest, AskResponse, Citation, DocumentInfo, HealthResponse
from app.services.ingestion import ALLOWED_EXTENSIONS, extract_pages, chunk_pages
from app.services.vector_store import get_vector_store
from app.workflows.graph import build_graph

router = APIRouter()
_graph = None

def workflow_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

@router.get("/health", response_model=HealthResponse)
def health():
    settings = get_settings()
    return HealthResponse(status="ok", model=settings.ollama_model, indexed_documents=get_vector_store().count_documents())

@router.get("/documents", response_model=list[DocumentInfo])
def documents():
    return [DocumentInfo(**d) for d in get_vector_store().list_documents()]

@router.post("/documents", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    settings = get_settings()
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "Unsupported file type. Upload PDF, TXT, or Markdown.")
    content = await file.read()
    if not content:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit.")
    # Content hash gives stable identity for repeat uploads with identical content.
    document_id = hashlib.sha256(content).hexdigest()[:16]
    safe_path = settings.upload_dir / f"{document_id}{suffix}"
    safe_path.write_bytes(content)
    try:
        pages = extract_pages(filename, content)
        chunks = chunk_pages(pages)
        count = get_vector_store().add_document(document_id, filename, chunks)
    except (ValueError, UnicodeError) as exc:
        safe_path.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Document indexing failed: {type(exc).__name__}") from exc
    return DocumentInfo(document_id=document_id, filename=filename, chunks=count)

@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    graph = workflow_graph()
    result = graph.invoke({"question": request.question, "document_ids": request.document_ids, "workflow": []})
    retrieved = result.get("retrieved", [])
    citations = [Citation(
        document_id=item["document_id"], filename=item["filename"], chunk_id=item["chunk_id"],
        page=item.get("page"), score=float(item.get("rerank_score", item.get("score", 0.0))), text=item["text"]
    ) for item in retrieved]
    scores = [float(item.get("score", 0)) for item in retrieved]
    best = max(scores, default=0.0)
    confidence = "high" if result.get("grounded") and best >= 0.55 else "medium" if result.get("grounded") else "low"
    return AskResponse(answer=result.get("answer", "No answer generated."), citations=citations,
        rewritten_question=result.get("rewritten_question", request.question), grounded=bool(result.get("grounded")),
        confidence=confidence, workflow=result.get("workflow", []))
