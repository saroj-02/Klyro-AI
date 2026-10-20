from types import SimpleNamespace

import numpy as np

from app.services.rag import RAGService
from app.services.vector_store import VectorStore


class FakeModel:
    def encode(self, sentences, normalize_embeddings=True, convert_to_numpy=True):
        assert sentences
        assert normalize_embeddings
        assert convert_to_numpy
        return np.array([[1.0, 0.0]], dtype="float32")


class FakeIndex:
    ntotal = 6

    def __init__(self):
        self.requested_k = None

    def search(self, vector, k):
        assert vector.shape == (1, 2)
        self.requested_k = k
        scores = np.array([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]], dtype="float32")
        ids = np.array([[0, 1, 2, 3, 4, 5]], dtype="int64")
        return scores[:, :k], ids[:, :k]


def test_search_applies_document_filter_before_top_k():
    store = VectorStore.__new__(VectorStore)
    store.settings = SimpleNamespace(retrieval_top_k=4)
    store.model = FakeModel()
    store.index = FakeIndex()
    store.metadata = [
        {
            "document_id": f"other-{index}",
            "filename": "other.pdf",
            "chunk_id": f"other-{index}:0",
            "text": "Unrelated evidence",
            "vector_index": index,
        }
        for index in range(5)
    ] + [
        {
            "document_id": "selected",
            "filename": "selected.pdf",
            "chunk_id": "selected:0",
            "text": "Selected document evidence",
            "vector_index": 5,
        }
    ]

    results = store.search("question", document_ids=["selected"], top_k=1)

    assert store.index.requested_k == store.index.ntotal
    assert [item["document_id"] for item in results] == ["selected"]


def make_service(answer):
    service = RAGService.__new__(RAGService)

    def fake_chat(prompt, temperature=0.0):
        assert prompt
        assert temperature == 0.1
        return answer

    service._chat = fake_chat
    return service


def test_answer_rejects_uncited_claims():
    answer, grounded = make_service("The answer is 42.").answer(
        "What is the answer?",
        [{"filename": "doc.pdf", "chunk_id": "doc:0", "text": "Answer is 42."}],
    )

    assert not grounded
    assert answer == "The retrieved passages do not support a reliable answer to this question."


def test_answer_rejects_citations_not_in_retrieved_context():
    answer, grounded = make_service("The answer is 42. [Source 2]").answer(
        "What is the answer?",
        [{"filename": "doc.pdf", "chunk_id": "doc:0", "text": "Answer is 42."}],
    )

    assert not grounded
    assert "wasn't retrieved" in answer


def test_answer_accepts_valid_citations():
    answer, grounded = make_service("The answer is 42. [Source 1]").answer(
        "What is the answer?",
        [{"filename": "doc.pdf", "chunk_id": "doc:0", "text": "Answer is 42."}],
    )

    assert grounded
    assert answer.endswith("[Source 1]")


def test_retrieval_combines_original_and_rewritten_queries(monkeypatch):
    class FakeStore:
        def __init__(self):
            self.queries = []

        def search(self, query, document_ids, top_k):
            self.queries.append((query, document_ids, top_k))
            if query == "original":
                return [{
                    "chunk_id": "doc:0",
                    "filename": "doc.pdf",
                    "text": "Original match",
                    "score": 0.8,
                }]
            return [
                {
                    "chunk_id": "doc:0",
                    "filename": "doc.pdf",
                    "text": "Duplicate match",
                    "score": 0.7,
                },
                {
                    "chunk_id": "doc:1",
                    "filename": "doc.pdf",
                    "text": "Rewritten-query match",
                    "score": 0.6,
                },
            ]

    class FakeReranker:
        def __init__(self):
            self.pairs = None

        def predict(self, pairs):
            self.pairs = pairs
            return [0.9, 0.8]

    service = RAGService.__new__(RAGService)
    service.settings = SimpleNamespace(retrieval_top_k=4, rerank_top_k=2)
    service.store = FakeStore()
    reranker = FakeReranker()
    monkeypatch.setattr("app.services.rag.get_reranker", lambda: reranker)

    results = service.retrieve("original", ["selected-doc"], search_query="rewritten")

    assert [entry[0] for entry in service.store.queries] == [
        "original",
        "rewritten",
    ]
    assert all(
        entry[1] == ["selected-doc"] and entry[2] == 4
        for entry in service.store.queries
    )
    assert reranker.pairs == [
        ("original", "Original match"),
        ("original", "Rewritten-query match"),
    ]
    assert [item["chunk_id"] for item in results] == ["doc:0", "doc:1"]
    assert results[0]["score"] == 0.8
