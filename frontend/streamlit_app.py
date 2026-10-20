import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")
st.set_page_config(page_title="Klyro-AI", page_icon="📚", layout="wide")
st.title("📚 Klyro-AI")
st.caption("Ask questions over your documents. Answers should be grounded in retrieved evidence.")

indexed_document_ids = []
with st.sidebar:
    st.header("1. Add documents")
    files = st.file_uploader("Upload PDF, TXT or Markdown", type=["pdf", "txt", "md", "markdown"], accept_multiple_files=True)
    if st.button("Index documents", type="primary", disabled=not files):
        for uploaded in files:
            with st.spinner(f"Indexing {uploaded.name}..."):
                try:
                    response = requests.post(f"{API_URL}/documents", files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")}, timeout=180)
                    response.raise_for_status()
                    indexed_document = response.json()
                    indexed_document_ids.append(indexed_document["document_id"])
                    st.success(f"Indexed {uploaded.name} ({indexed_document['chunks']} chunks)")
                except requests.RequestException as exc:
                    if exc.response is not None:
                        try:
                            body = exc.response.json()
                        except ValueError:
                            body = exc.response.text
                        detail = body.get("detail") if isinstance(body, dict) else body
                        message = f"Could not index {uploaded.name}"
                        if detail:
                            message += f": {detail}"
                        message += f" (HTTP {exc.response.status_code})"
                        st.error(message)
                    else:
                        st.error(f"Could not index {uploaded.name}: {exc}")
    st.divider()
    st.header("2. Indexed documents")
    try:
        docs_response = requests.get(f"{API_URL}/documents", timeout=10)
        docs_response.raise_for_status()
        docs = docs_response.json()
        document_ids = [document["document_id"] for document in docs]
        if indexed_document_ids:
            st.session_state["selected_document_ids"] = list(dict.fromkeys(indexed_document_ids))
        elif "selected_document_ids" not in st.session_state:
            st.session_state["selected_document_ids"] = document_ids[-1:]
        else:
            st.session_state["selected_document_ids"] = [
                document_id
                for document_id in st.session_state["selected_document_ids"]
                if document_id in document_ids
            ]
        search_all_documents = st.checkbox(
            "Search all indexed documents",
            value=False,
            help="When off, questions use only the selected documents.",
        )
        selected = st.multiselect(
            "Search within these documents",
            options=document_ids,
            format_func=lambda document_id: next(
                f"{document['filename']} · {document['chunks']} chunks"
                for document in docs
                if document["document_id"] == document_id
            ),
            disabled=search_all_documents or not docs,
            key="selected_document_ids",
        )
        selected_document_ids = (
            [] if search_all_documents
            else selected
        )
    except requests.RequestException:
        docs = []
        selected = []
        selected_document_ids = []
        st.warning("Backend is unavailable. Start the API first.")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander("Retrieved evidence and citations"):
                for i, citation in enumerate(message["citations"], 1):
                    st.markdown(f"**[{i}] {citation['filename']}** · page {citation.get('page') or 'N/A'} · score {citation['score']:.3f}")
                    st.write(citation["text"])

question = st.chat_input("Ask a question about your indexed documents...")
if question:
    if not selected_document_ids:
        st.error("Select at least one indexed document or enable 'Search all indexed documents'.")
        st.stop()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching, reranking and checking evidence..."):
            try:
                response = requests.post(f"{API_URL}/ask", json={"question": question, "document_ids": selected_document_ids}, timeout=240)
                response.raise_for_status()
                data = response.json()
                st.markdown(data["answer"])
                st.caption(f"Grounded: {data['grounded']} · Confidence indicator: {data['confidence']} · Query: {data['rewritten_question']}")
                with st.expander("Workflow trace"):
                    st.write(" → ".join(data["workflow"]))
                with st.expander("Retrieved evidence and citations"):
                    if not data["citations"]:
                        st.info("No evidence was retrieved.")
                    for i, citation in enumerate(data["citations"], 1):
                        st.markdown(f"**[{i}] {citation['filename']}** · page {citation.get('page') or 'N/A'} · score {citation['score']:.3f}")
                        st.write(citation["text"])
                st.session_state.messages.append({"role": "assistant", "content": data["answer"], "citations": data["citations"]})
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")
