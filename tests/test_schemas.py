import pytest
from pydantic import ValidationError
from app.models import AskRequest

def test_ask_request_accepts_question():
    assert AskRequest(question="What does this document say?").document_ids == []

def test_ask_request_rejects_short_question():
    with pytest.raises(ValidationError):
        AskRequest(question="hi")
