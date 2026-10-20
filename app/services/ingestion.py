
import io
import re
from pathlib import Path

import fitz
import pytesseract
from PIL import Image


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


def extract_pages(filename: str, content: bytes) -> list[dict]:
    """Extract readable text from PDF, TXT and Markdown files."""
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    if not content:
        raise ValueError("The uploaded document is empty.")

    if extension == ".pdf":
        return extract_pdf_with_ocr(content)

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Text file must use UTF-8 encoding."
        ) from exc

    text = normalize_text(text)

    if not text:
        raise ValueError("No readable text found in the uploaded document.")

    return [{"page": 1, "text": text}]


def extract_pdf_with_ocr(pdf_bytes: bytes) -> list[dict]:
    """Extract PDF text and use OCR for pages with no embedded text."""
    pages = []

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if not text:
                    pixmap = page.get_pixmap(
                        matrix=fitz.Matrix(2, 2),
                        alpha=False,
                    )
                    image = Image.open(
                        io.BytesIO(pixmap.tobytes("png"))
                    ).convert("RGB")

                    text = pytesseract.image_to_string(
                        image,
                        lang="eng",
                    ).strip()

                text = normalize_text(text)

                if text:
                    pages.append({
                        "page": page_number,
                        "text": text,
                    })

    except Exception as exc:
        raise ValueError(
            f"Could not read PDF: {exc}"
        ) from exc

    if not pages:
        raise ValueError(
            "No readable text found. The PDF may be scanned, "
            "blank, encrypted, or require a different OCR language."
        )

    return pages


def normalize_text(text: str) -> str:
    """Normalize whitespace without removing meaningful paragraph breaks."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split extracted page text into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []

    for page_data in pages:
        text = normalize_text(page_data.get("text", ""))

        if not text:
            continue

        page_number = page_data.get(
            "page",
            page_data.get("page_number", 1),
        )

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page_number,
                    "page_number": page_number,
                })

            if end == len(text):
                break

            start = end - overlap

    return chunks
