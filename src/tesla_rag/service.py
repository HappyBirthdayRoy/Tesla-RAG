from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any

from tesla_rag.config import DEFAULT_COLLECTION, DEFAULT_OPENAI_BASE_URL, DEFAULT_OPENAI_MODEL, DEFAULT_TOP_K
from tesla_rag.ingest import extract_chunks
from tesla_rag.models import Citation
from tesla_rag.vectorstore import EmbeddingFn, VectorStore


def _missing_openai_key_error() -> RuntimeError:
    return RuntimeError(
        "OPENAI_API_KEY is required for answer synthesis. "
        "Set OPENAI_API_KEY and optionally TESLA_RAG_OPENAI_MODEL/TESLA_RAG_OPENAI_BASE_URL, "
        "then retry. Fallback: use retrieve() for context-only output until credentials are configured."
    )


class RagService:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_fn: EmbeddingFn | None = None,
        llm_client: Any | None = None,
        llm_model: str | None = None,
    ) -> None:
        self.store = VectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_fn=embedding_fn,
        )
        self._llm_client = llm_client
        self._llm_model = llm_model or DEFAULT_OPENAI_MODEL

    def ingest(self, pdf_paths: list[str]) -> int:
        chunks = extract_chunks(pdf_paths)
        return self.store.upsert_chunks(chunks)

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        response = self.store.query(question=question, top_k=top_k)
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

    def _ensure_llm_client(self) -> Any:
        if self._llm_client is not None:
            return self._llm_client

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise _missing_openai_key_error()

        from openai import OpenAI

        base_url = DEFAULT_OPENAI_BASE_URL or None
        self._llm_client = OpenAI(api_key=api_key, base_url=base_url)
        return self._llm_client

    def _synthesize_answer(self, question: str, contexts: list[dict]) -> str:
        client = self._ensure_llm_client()
        context_lines = []
        for i, ctx in enumerate(contexts, start=1):
            text = ctx["text"].strip().replace("\n", " ")
            context_lines.append(f"[{i}] {ctx['source_file']} p.{ctx['page']}: {text}")

        prompt = (
            "Answer the user question using only the provided contexts from Tesla PDFs. "
            "If the answer is not in the contexts, say exactly: Not found in provided contexts.\n\n"
            f"Question: {question}\n\n"
            "Contexts:\n"
            + "\n".join(context_lines)
            + "\n\nReturn only the answer text."
        )

        response = client.responses.create(
            model=self._llm_model,
            input=prompt,
            temperature=0,
        )
        output_text = (response.output_text or "").strip()
        if not output_text:
            return "Not found in provided contexts."
        return output_text

    def answer(self, question: str, top_k: int = DEFAULT_TOP_K) -> dict:
        contexts = self.retrieve(question=question, top_k=top_k)
        if not contexts:
            return {
                "answer": "I could not find relevant information in the indexed PDFs.",
                "citations": [],
                "contexts": [],
            }

        citations_map: OrderedDict[tuple[str, int], Citation] = OrderedDict()
        for ctx in contexts:
            key = (ctx["source_file"], ctx["page"])
            citations_map[key] = Citation(source_file=ctx["source_file"], page=ctx["page"])

        answer_text = self._synthesize_answer(question=question, contexts=contexts)
        citations = [{"source_file": c.source_file, "page": c.page} for c in citations_map.values()]

        return {
            "answer": answer_text,
            "citations": citations,
            "contexts": contexts,
        }
