from __future__ import annotations

from collections import OrderedDict

from tesla_rag.config import DEFAULT_COLLECTION, DEFAULT_TOP_K
from tesla_rag.ingest import extract_chunks
from tesla_rag.models import Citation
from tesla_rag.vectorstore import EmbeddingFn, VectorStore


class RagService:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_fn: EmbeddingFn | None = None,
    ) -> None:
        self.store = VectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_fn=embedding_fn,
        )

    def ingest(self, pdf_paths: list[str]) -> int:
        chunks = extract_chunks(pdf_paths)
        return self.store.upsert_chunks(chunks)

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        response = self.store.hybrid_query(question=question, top_k=top_k)
        docs = response.get("documents", [[]])[0]
        metas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        out: list[dict] = []
        for i, doc in enumerate(docs):
            md = metas[i] if i < len(metas) else {}
            dist = distances[i] if i < len(distances) else None
            out.append(
                {
                    "text": doc,
                    "source_file": md.get("source_file", "unknown"),
                    "page": int(md.get("page", 0)),
                    "section": md.get("section", "General"),
                    "distance": dist,
                }
            )
        return out

    def answer(self, question: str, top_k: int = DEFAULT_TOP_K) -> dict:
        contexts = self.retrieve(question=question, top_k=top_k)
        if not contexts:
            return {
                "answer": "I could not find relevant information in the indexed PDFs.",
                "citations": [],
                "contexts": [],
            }

        citations_map: OrderedDict[tuple[str, int], Citation] = OrderedDict()
        snippets: list[str] = []
        for ctx in contexts:
            key = (ctx["source_file"], ctx["page"])
            citations_map[key] = Citation(source_file=ctx["source_file"], page=ctx["page"])
            snippet = ctx["text"].strip().replace("\n", " ")
            snippet = snippet[:240] + ("..." if len(snippet) > 240 else "")
            snippets.append(f"[{ctx['source_file']} p.{ctx['page']}] {snippet}")

        answer_text = "\n".join(snippets)
        citations = [{"source_file": c.source_file, "page": c.page} for c in citations_map.values()]

        return {
            "answer": answer_text,
            "citations": citations,
            "contexts": contexts,
        }
