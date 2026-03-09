from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from tesla_rag.config import DEFAULT_CHUNK_MAX_CHARS, DEFAULT_CHUNK_OVERLAP_CHARS
from tesla_rag.models import Chunk

_HEADING_RE = re.compile(r"^(\d+(\.\d+)*\s+.+|[A-Z][A-Z\s\-:]{4,})$")
_PERIOD_RE = re.compile(r"\b(?:Q[1-4]-\d{4}|\d{4}|YoY)\b", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"^\(?-?\d[\d,]*(?:\.\d+)?%?\)?$")


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _period_key(period: str) -> str:
    return period.lower().replace("-", " ")


def _strip_row_suffix_markers(label: str) -> str:
    value = label.strip()
    while True:
        nxt = re.sub(r"\s*\(\d+\)\s*$", "", value)
        if nxt == value:
            return value.strip()
        value = nxt


def _is_numeric_token(token: str) -> bool:
    return bool(_NUMERIC_RE.match(token))


def _extract_row_alias(line: str, periods: list[str]) -> str | None:
    if len(periods) < 4:
        return None

    tokens = line.split()
    if len(tokens) < len(periods) + 2:
        return None

    values_rev: list[str] = []
    i = len(tokens) - 1
    while i >= 0 and len(values_rev) < len(periods):
        tok = tokens[i]
        low = tok.lower()
        if low == "bp" and i > 0 and _is_numeric_token(tokens[i - 1]):
            values_rev.append(f"{tokens[i - 1]} bp")
            i -= 2
            continue
        if _is_numeric_token(tok):
            values_rev.append(tok)
            i -= 1
            continue
        if values_rev:
            break
        i -= 1

    if len(values_rev) != len(periods):
        return None

    row_label = _strip_row_suffix_markers(" ".join(tokens[: i + 1]))
    if len(row_label) < 4:
        return None
    if row_label[0].isdigit() or row_label.startswith("("):
        return None

    values = list(reversed(values_rev))
    pairs = [f"{_period_key(period)} {value}" for period, value in zip(periods, values)]
    return f"table_row {row_label.lower()} | " + " | ".join(pairs)


def enrich_table_row_aliases(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    out_lines: list[str] = []
    active_periods: list[str] = []

    for line in lines:
        out_lines.append(line)

        periods_in_line = _PERIOD_RE.findall(line)
        if len(periods_in_line) >= 4:
            active_periods = periods_in_line
            continue

        if not active_periods:
            continue

        alias = _extract_row_alias(line, active_periods)
        if alias:
            out_lines.append(alias)

    return clean_text("\n".join(out_lines))


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


def chunk_text(
    section_title: str,
    section_text: str,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[tuple[str, str]]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be < max_chars")

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


def extract_chunks_from_pdf(
    pdf_path: str,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
    reader = PdfReader(pdf_path)
    source_file = Path(pdf_path).name
    results: list[Chunk] = []

    for page_index, page in enumerate(reader.pages):
        page_text = clean_text(page.extract_text() or "")
        if not page_text:
            continue

        page_text = enrich_table_row_aliases(page_text)
        sections = split_sections(page_text)
        for section_i, (section_title, section_text) in enumerate(sections):
            for chunk_i, (_, text) in enumerate(
                chunk_text(section_title, section_text, max_chars=max_chars, overlap_chars=overlap_chars)
            ):
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


def extract_chunks(
    pdf_paths: Iterable[str],
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in pdf_paths:
        all_chunks.extend(extract_chunks_from_pdf(path, max_chars=max_chars, overlap_chars=overlap_chars))
    return all_chunks
