from __future__ import annotations

from pathlib import Path

from tesla_rag.ingest import clean_text, extract_chunks_from_pdf, split_sections


PDF_A = "/home/node/.openclaw/media/inbound/0af5279c-11f2-4069-8708-061dfda248ae.pdf"


def test_clean_text_basic() -> None:
    assert clean_text("A   B\n\n\nC") == "A B\n\nC"


def test_split_sections_returns_content() -> None:
    sections = split_sections("INTRODUCTION\nLine 1\nLine 2")
    assert sections


def test_extract_chunks_from_real_pdf_smoke() -> None:
    assert Path(PDF_A).exists(), "Expected inbound PDF to exist"
    chunks = extract_chunks_from_pdf(PDF_A)
    assert len(chunks) > 0
    assert chunks[0].source_file.endswith(".pdf")
    assert chunks[0].page >= 1
