# 3–5 minute demo video outline

**0:00–0:25 — Introduction**
“Hi, I built Klyro-AI, an advanced RAG assistant that answers questions over PDF, TXT and Markdown files. It separates ingestion, retrieval, orchestration, generation, and the user interface.”

**0:25–1:05 — Architecture**
Show the README architecture. Explain FastAPI/Pydantic, Streamlit, Sentence Transformers, FAISS, cross-encoder reranking, LangGraph, and Ollama.

**1:05–2:00 — Upload and index**
Upload a sample document. Show indexed chunk count and document list. Explain page-aware chunking and vector metadata.

**2:00–3:10 — Grounded Q&A**
Ask a question answered in the document. Expand citations/evidence and workflow trace. Ask an unanswerable question and show abstention. If possible, select one document and show document-scoped retrieval.

**3:10–4:00 — Engineering details**
Show API docs and Pydantic models. Briefly explain input validation, query rewriting, reranking, evidence gate, and tests.

**4:00–4:30 — Limitations and close**
Explain that confidence is a heuristic, the current local index is not multi-tenant, and production would need auth, monitoring, a verifier, OCR, and deployment hardening. Close by showing the GitHub repository.

Record your own screen and voice. Do not claim a feature works unless you have tested it live.
