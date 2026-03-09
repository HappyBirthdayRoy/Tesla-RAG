from __future__ import annotations

import os
from pathlib import Path

PDF_A = "/home/node/.openclaw/media/inbound/0af5279c-11f2-4069-8708-061dfda248ae.pdf"
PDF_B = "/home/node/.openclaw/media/inbound/0e52d151-9288-4c1f-893e-f4d10ec0e029.pdf"
ALLOWED_PDFS = (PDF_A, PDF_B)

DEFAULT_CHROMA_DIR = os.getenv("TESLA_RAG_CHROMA_DIR", ".chroma")
DEFAULT_COLLECTION = os.getenv("TESLA_RAG_COLLECTION", "tesla_docs")
DEFAULT_TOP_K = int(os.getenv("TESLA_RAG_TOP_K", "4"))
DEFAULT_CHUNK_MAX_CHARS = int(os.getenv("TESLA_RAG_CHUNK_MAX_CHARS", "1000"))
DEFAULT_CHUNK_OVERLAP_CHARS = int(os.getenv("TESLA_RAG_CHUNK_OVERLAP_CHARS", "200"))


def configured_pdf_paths() -> list[str]:
    env_value = os.getenv("TESLA_RAG_PDF_PATHS", "")
    if not env_value.strip():
        return list(ALLOWED_PDFS)
    return [p.strip() for p in env_value.split(",") if p.strip()]


def validate_pdf_paths(paths: list[str]) -> list[str]:
    allowed = set(ALLOWED_PDFS)
    normalized = []
    for path in paths:
        if path not in allowed and os.getenv("TESLA_RAG_ALLOW_NONDEFAULT", "0") != "1":
            raise ValueError(f"Unsupported data source: {path}")
        if not Path(path).exists():
            raise FileNotFoundError(path)
        normalized.append(path)
    return normalized
