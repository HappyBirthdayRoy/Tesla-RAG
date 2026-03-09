from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tesla_rag.config import DEFAULT_CHROMA_DIR, DEFAULT_TOP_K
from tesla_rag.service import RagService


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=10)


class CitationOut(BaseModel):
    source_file: str
    page: int


class ContextOut(BaseModel):
    text: str
    source_file: str
    page: int
    section: str
    distance: float | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    contexts: list[ContextOut]


app = FastAPI(title="tesla-rag", version="0.1.0")
_service: RagService | None = None


def get_service() -> RagService:
    global _service
    if _service is None:
        _service = RagService(persist_dir=DEFAULT_CHROMA_DIR)
    return _service


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    service = get_service()
    if service.store.count() == 0:
        raise HTTPException(status_code=400, detail="Index is empty. Run ingestion first.")

    result = service.answer(question=payload.question, top_k=payload.top_k)
    return AskResponse(**result)
