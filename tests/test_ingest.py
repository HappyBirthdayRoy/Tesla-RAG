from __future__ import annotations

from pathlib import Path

import pytest

from tesla_rag.ingest import (
    chunk_text,
    clean_text,
    enrich_table_row_aliases,
    extract_chunks_from_pdf,
    split_sections,
)


PDF_A = "/home/node/.openclaw/media/inbound/0af5279c-11f2-4069-8708-061dfda248ae.pdf"


def test_clean_text_basic() -> None:
    assert clean_text("A   B\n\n\nC") == "A B\n\nC"


def test_split_sections_returns_content() -> None:
    sections = split_sections("INTRODUCTION\nLine 1\nLine 2")
    assert sections


def test_enrich_table_row_aliases_adds_period_value_aliases() -> None:
    text = (
        "($ in millions) Q3-2024 Q4-2024 Q1-2025 Q2-2025 Q3-2025 YoY\n"
        "Total revenues 25,182 25,707 19,335 22,496 28,095 12%\n"
    )
    enriched = enrich_table_row_aliases(text)
    assert "table_row total revenues" in enriched
    assert "q3 2025 28,095" in enriched
    assert "yoy 12%" in enriched


def test_extract_chunks_from_real_pdf_smoke() -> None:
    assert Path(PDF_A).exists(), "Expected inbound PDF to exist"
    chunks = extract_chunks_from_pdf(PDF_A)
    assert len(chunks) > 0
    assert chunks[0].source_file.endswith(".pdf")
    assert chunks[0].page >= 1


def test_chunk_text_validates_overlap_lt_max() -> None:
    with pytest.raises(ValueError):
        chunk_text("General", "A" * 1200, max_chars=1000, overlap_chars=1000)
