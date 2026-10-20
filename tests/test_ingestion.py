from app.services.ingestion import chunk_pages, normalize_text

def test_normalize_text_removes_extra_whitespace():
    assert normalize_text(" hello   world\n\n\nnext ") == "hello world\n\nnext"

def test_chunking_preserves_content_and_page():
    text = "A" * 250 + ". " + "B" * 250
    chunks = chunk_pages([{"page": 3, "text": text}], chunk_size=180, overlap=30)
    assert len(chunks) >= 2
    assert all(c["page"] == 3 for c in chunks)
    assert all(c["text"] for c in chunks)

def test_chunking_rejects_invalid_overlap():
    try:
        chunk_pages([(None, "hello")], chunk_size=100, overlap=100)
        assert False, "expected ValueError"
    except ValueError:
        pass
