from __future__ import annotations

from tesla_rag.service import RagService


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
