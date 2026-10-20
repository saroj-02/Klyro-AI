from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from app.services.rag import RAGService
from app.core.config import get_settings

class RAGState(TypedDict, total=False):
    question: str
    document_ids: list[str]
    rewritten_question: str
    retrieved: list[dict[str, Any]]
    answer: str
    grounded: bool
    workflow: list[str]


def build_graph():
    service = RAGService()
    settings = get_settings()

    def rewrite(state: RAGState) -> RAGState:
        rewritten = service.rewrite_query(state["question"])
        return {"rewritten_question": rewritten, "workflow": state.get("workflow", []) + ["query_rewrite"]}

    def retrieve(state: RAGState) -> RAGState:
        docs = service.retrieve(
            state["question"],
            state.get("document_ids", []),
            search_query=state.get("rewritten_question"),
        )
        return {"retrieved": docs, "workflow": state.get("workflow", []) + ["vector_search_and_rerank"]}

    def evaluate(state: RAGState) -> RAGState:
        docs = state.get("retrieved", [])
        best = max((d.get("score", 0.0) for d in docs), default=0.0)
        # Normalized cosine similarity is expected in [-1,1]; the threshold is configurable.
        enough = bool(docs) and best >= settings.min_relevance_score
        return {"grounded": enough, "workflow": state.get("workflow", []) + ["evidence_gate"]}

    def generate(state: RAGState) -> RAGState:
        if not state.get("grounded", False):
            answer = "I couldn't find sufficiently relevant evidence in the selected documents. Try rephrasing the question or selecting another document."
            grounded = False
        else:
            answer, grounded = service.answer(state["question"], state.get("retrieved", []))
        return {"answer": answer, "grounded": grounded, "workflow": state.get("workflow", []) + ["grounded_generation"]}

    graph = StateGraph(RAGState)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve)
    graph.add_node("evaluate", evaluate)
    graph.add_node("generate", generate)
    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "evaluate")
    graph.add_edge("evaluate", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
