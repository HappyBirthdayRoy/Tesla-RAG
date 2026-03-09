from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    source_file: str
    page: int
    section: str


@dataclass(slots=True)
class Citation:
    source_file: str
    page: int
