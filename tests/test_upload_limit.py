import asyncio
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api import routes
from app.core.config import Settings


def test_default_upload_limit_is_100_mb(monkeypatch):
    monkeypatch.delenv("MAX_UPLOAD_MB", raising=False)
    assert Settings(_env_file=None).max_upload_mb == 100


def test_upload_rejects_files_above_configured_limit(monkeypatch, tmp_path):
    settings = SimpleNamespace(max_upload_mb=1, upload_dir=tmp_path)
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    upload = UploadFile(
        filename="Numpy.pdf",
        file=BytesIO(b"x" * (1024 * 1024 + 1)),
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.upload_document(upload))

    assert error.value.status_code == 413
    assert error.value.detail == "File exceeds 1 MB limit."
