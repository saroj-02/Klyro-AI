import json
import threading
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings

class VectorStore:
    """A small persistent FAISS index with metadata stored beside it.

    For a multi-user deployment, replace this process-local store with a tenant-aware
    database/vector service and enforce ownership at the API boundary.
    """
    def __init__(self):
        self.settings = get_settings()
        self.model = SentenceTransformer(self.settings.embedding_model)
        self.lock = threading.RLock()
        self.index_path = self.settings.index_dir / "documents.faiss"
        self.meta_path = self.settings.index_dir / "metadata.json"
        self.metadata: list[dict] = []
        self.index = None
        self._load()

    def _load(self):
        with self.lock:
            if self.index_path.exists() and self.meta_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def _save(self):
        self.settings.index_dir.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_document(self, document_id: str, filename: str, chunks: list[dict]) -> int:
        if not chunks:
            raise ValueError("No readable text found in the uploaded document")
        texts = [item["text"] for item in chunks]
        vectors = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        with self.lock:
            # Re-ingestion replaces the previous version of this document.
            keep = [m for m in self.metadata if m["document_id"] != document_id]
            if len(keep) != len(self.metadata):
                keep_vectors = []
                if self.index is not None:
                    for i, m in enumerate(self.metadata):
                        if m["document_id"] != document_id:
                            keep_vectors.append(self.index.reconstruct(i))
                self.metadata = keep
                self.index = None
                if keep_vectors:
                    self.index = faiss.IndexFlatIP(vectors.shape[1])
                    self.index.add(np.asarray(keep_vectors, dtype="float32"))
            if self.index is None:
                self.index = faiss.IndexFlatIP(vectors.shape[1])
            if self.index.d != vectors.shape[1]:
                raise RuntimeError("Embedding dimension mismatch; clear the old index after changing embedding models")
            start_idx = len(self.metadata)
            self.index.add(vectors)
            for i, item in enumerate(chunks):
                self.metadata.append({
                    "document_id": document_id, "filename": filename,
                    "chunk_id": f"{document_id}:{i}", "page": item.get("page"),
                    "text": item["text"], "vector_index": start_idx + i,
                })
            self._save()
        return len(chunks)

    def search(self, query: str, document_ids: list[str] | None = None, top_k: int | None = None) -> list[dict]:
        if not self.metadata or self.index is None:
            return []
        settings = self.settings
        vector = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        allowed = set(document_ids or [])
        # Filtering a small global top-k can hide relevant chunks from a selected document.
        k = self.index.ntotal if allowed else min(
            max((top_k or settings.retrieval_top_k) * 5, top_k or settings.retrieval_top_k),
            self.index.ntotal,
        )
        scores, ids = self.index.search(vector, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            meta = self.metadata[int(idx)]
            if allowed and meta["document_id"] not in allowed:
                continue
            results.append({**meta, "score": float(score)})
            if len(results) >= (top_k or settings.retrieval_top_k):
                break
        return results

    def list_documents(self) -> list[dict]:
        docs = {}
        for item in self.metadata:
            docs[item["document_id"]] = {"document_id": item["document_id"], "filename": item["filename"], "chunks": docs.get(item["document_id"], {}).get("chunks", 0) + 1}
        return list(docs.values())

    def count_documents(self) -> int:
        return len({m["document_id"] for m in self.metadata})

_store: VectorStore | None = None
_store_lock = threading.Lock()
def get_vector_store() -> VectorStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = VectorStore()
        return _store
