from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from tesla_rag.models import Chunk

_HEADING_RE = re.compile(r"^(\d+(\.\d+)*\s+.+|[A-Z][A-Z\s\-:]{4,}|[A-Z][\w\s\-:]{3,})$")


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_heading(line: str) -> bool:
    line = line.strip()
    if len(line) < 4 or len(line) > 120:
        return False
    if line.endswith("."):
        return False
    return bool(_HEADING_RE.match(line))


def split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = "General"
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue

        if is_heading(line):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line
            current_lines = []
            continue

        current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    output: list[tuple[str, str]] = []
    for title, lines in sections:
        section_text = clean_text("\n".join(lines))
        if section_text:
            output.append((title, section_text))
    return output


def chunk_text(section_title: str, section_text: str, max_chars: int = 1000, overlap_chars: int = 200) -> list[tuple[str, str]]:
    if len(section_text) <= max_chars:
        return [(section_title, section_text)]

    chunks: list[tuple[str, str]] = []
    start = 0
    while start < len(section_text):
        end = min(start + max_chars, len(section_text))
        piece = section_text[start:end].strip()
        if piece:
            chunks.append((section_title, piece))
        if end == len(section_text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def extract_chunks_from_pdf(pdf_path: str) -> list[Chunk]:
    reader = PdfReader(pdf_path)
    source_file = Path(pdf_path).name
    results: list[Chunk] = []

    for page_index, page in enumerate(reader.pages):
        page_text = clean_text(page.extract_text() or "")
        if not page_text:
            continue

        sections = split_sections(page_text)
        for section_i, (section_title, section_text) in enumerate(sections):
            for chunk_i, (_, text) in enumerate(chunk_text(section_title, section_text)):
                chunk_id = f"{source_file}-p{page_index+1}-s{section_i}-c{chunk_i}"
                results.append(
                    Chunk(
                        id=chunk_id,
                        text=text,
                        source_file=source_file,
                        page=page_index + 1,
                        section=section_title,
                    )
                )
    return results


def extract_chunks(pdf_paths: Iterable[str]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in pdf_paths:
        all_chunks.extend(extract_chunks_from_pdf(path))
    return all_chunks
