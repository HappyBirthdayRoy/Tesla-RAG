from __future__ import annotations

import pytest

from tesla_rag.service import RagService
from tesla_rag.vectorstore import VectorStore


def test_retrieve_smoke(tmp_path, fake_embedding) -> None:
    svc = RagService(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_collection",
        embedding_fn=fake_embedding,
    )

    inserted = svc.ingest(
        [
            "/home/node/.openclaw/media/inbound/0af5279c-11f2-4069-8708-061dfda248ae.pdf",
            "/home/node/.openclaw/media/inbound/0e52d151-9288-4c1f-893e-f4d10ec0e029.pdf",
        ]
    )
    assert inserted > 0

    results = svc.retrieve("What is the main topic?", top_k=3)
    assert len(results) > 0
    assert {"text", "source_file", "page"}.issubset(results[0].keys())


def test_hybrid_query_fuses_vector_and_bm25_rankings() -> None:
    store = VectorStore.__new__(VectorStore)

    def fake_vector_query(question: str, top_k: int) -> dict:
        return {
            "ids": [["x", "y", "z", "a"]],
            "documents": [["alpha text", "beta text", "gamma text", "delta text"]],
            "metadatas": [[
                {"source_file": "a.pdf", "page": 1, "section": "S"},
                {"source_file": "a.pdf", "page": 2, "section": "S"},
                {"source_file": "a.pdf", "page": 3, "section": "S"},
                {"source_file": "a.pdf", "page": 4, "section": "S"},
            ]],
            "distances": [[0.01, 0.02, 0.03, 0.04]],
        }

    def fake_all_chunks() -> dict:
        return {
            "ids": ["x", "y", "z", "a", "c"],
            "documents": ["alpha text", "beta text", "gamma text", "delta text", "keyword only hit"],
            "metadatas": [
                {"source_file": "a.pdf", "page": 1, "section": "S"},
                {"source_file": "a.pdf", "page": 2, "section": "S"},
                {"source_file": "a.pdf", "page": 3, "section": "S"},
                {"source_file": "a.pdf", "page": 4, "section": "S"},
                {"source_file": "b.pdf", "page": 9, "section": "S"},
            ],
        }

    store.query = fake_vector_query  # type: ignore[method-assign]
    store.get_all_chunks = fake_all_chunks  # type: ignore[method-assign]

    out = store.hybrid_query("keyword", top_k=2, rrf_k=60)
    docs = out["documents"][0]
    assert "keyword only hit" in docs
    assert len(docs) == 2


def test_answer_fails_fast_when_openai_key_missing(tmp_path, fake_embedding, monkeypatch) -> None:
    svc = RagService(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_collection_failfast",
        embedding_fn=fake_embedding,
    )
    inserted = svc.ingest(
        [
            "/home/node/.openclaw/media/inbound/0af5279c-11f2-4069-8708-061dfda248ae.pdf",
            "/home/node/.openclaw/media/inbound/0e52d151-9288-4c1f-893e-f4d10ec0e029.pdf",
        ]
    )
    assert inserted > 0
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required for answer synthesis"):
        svc.answer("What was total revenue in Q3 2025?", top_k=2)
