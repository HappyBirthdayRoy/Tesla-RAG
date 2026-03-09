from __future__ import annotations

import os
import re
from collections import OrderedDict
from typing import Any

from tesla_rag.config import DEFAULT_ANTHROPIC_MODEL, DEFAULT_COLLECTION, DEFAULT_TOP_K
from tesla_rag.ingest import extract_chunks
from tesla_rag.models import Citation
from tesla_rag.vectorstore import EmbeddingFn, VectorStore


def _missing_anthropic_key_error() -> RuntimeError:
    return RuntimeError(
        "ANTHROPIC_API_KEY is required for answer synthesis. "
        "Set ANTHROPIC_API_KEY and optionally TESLA_RAG_ANTHROPIC_MODEL, "
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
        self._llm_model = llm_model or DEFAULT_ANTHROPIC_MODEL

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

    def _ensure_llm_client(self) -> Any:
        if self._llm_client is not None:
            return self._llm_client

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise _missing_anthropic_key_error()

        from anthropic import Anthropic

        self._llm_client = Anthropic(api_key=api_key)
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

        response = client.messages.create(
            model=self._llm_model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        output_text = (response.content[0].text if response.content else "").strip()
        if not output_text:
            return "Not found in provided contexts."
        return output_text

    @staticmethod
    def _extract_number_candidates(text: str) -> list[tuple[str, int, int]]:
        candidates: list[tuple[str, int, int]] = []
        pattern = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\b")
        for match in pattern.finditer(text):
            token = match.group(0)
            value = int(token.replace(",", ""))
            if 1900 <= value <= 2100:
                continue
            candidates.append((token, match.start(), match.end()))
        return candidates

    @staticmethod
    def _question_keywords(question: str) -> set[str]:
        words = re.findall(r"[a-z]+", question.lower())
        stop = {
            "in",
            "the",
            "what",
            "is",
            "and",
            "of",
            "to",
            "for",
            "on",
            "from",
            "with",
            "q",
            "fy",
            "usd",
            "millions",
            "million",
            "gaap",
        }
        return {w for w in words if len(w) >= 4 and w not in stop}

    def _select_best_numeric_answer(self, question: str, answer_text: str, contexts: list[dict]) -> str:
        if not re.search(r"\d", question):
            return answer_text

        question_keywords = self._question_keywords(question)
        scored: list[tuple[int, str]] = []

        def add_candidates(source_text: str, source_bias: int) -> None:
            for token, start, end in self._extract_number_candidates(source_text):
                window = source_text[max(0, start - 100) : min(len(source_text), end + 100)].lower()
                keyword_score = sum(1 for kw in question_keywords if kw in window) * 10
                comma_bonus = 5 if "," in token else 0
                score = source_bias + keyword_score + comma_bonus
                scored.append((score, token))

        add_candidates(answer_text, 100)
        for ctx in contexts:
            add_candidates(ctx.get("text", ""), 0)

        if not scored:
            return answer_text
        scored.sort(key=lambda row: row[0], reverse=True)
        return scored[0][1]

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
        answer_text = self._select_best_numeric_answer(question=question, answer_text=answer_text, contexts=contexts)
        citations = [{"source_file": c.source_file, "page": c.page} for c in citations_map.values()]

        return {
            "answer": answer_text,
            "citations": citations,
            "contexts": contexts,
        }
