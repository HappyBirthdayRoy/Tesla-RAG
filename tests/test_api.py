from __future__ import annotations

from fastapi.testclient import TestClient

from tesla_rag import api
from tesla_rag.service import RagService


PDFS = [
    "/home/node/.openclaw/media/inbound/0af5279c-11f2-4069-8708-061dfda248ae.pdf",
    "/home/node/.openclaw/media/inbound/0e52d151-9288-4c1f-893e-f4d10ec0e029.pdf",
]


def test_ask_endpoint_smoke(tmp_path, fake_embedding) -> None:
    test_service = RagService(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="api_test_collection",
        embedding_fn=fake_embedding,
    )
    inserted = test_service.ingest(PDFS)
    assert inserted > 0

    api._service = test_service
    client = TestClient(api.app)

    response = client.post("/ask", json={"question": "Summarize key points"})
    assert response.status_code == 200
    payload = response.json()
    assert "answer" in payload
    assert len(payload["citations"]) > 0
    assert len(payload["contexts"]) > 0
