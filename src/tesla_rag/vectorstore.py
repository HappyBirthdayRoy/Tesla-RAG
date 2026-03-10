from __future__ import annotations

import re
from typing import Any, Protocol

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi

from tesla_rag.models import Chunk

_STOP_WORDS = {
    "and",
    "for",
    "from",
    "in",
    "is",
    "millions",
    "million",
    "of",
    "on",
    "the",
    "to",
    "usd",
    "what",
}
_TABLE_ROW_RE = re.compile(r"table_row\s+([^|]+)\|", re.IGNORECASE)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _keyword_tokens(text: str) -> set[str]:
    tokens = _normalize_text(text).split()
    return {token for token in tokens if len(token) >= 3 and token not in _STOP_WORDS}


def _table_row_bonus(question: str, document: str) -> float:
    question_tokens = _keyword_tokens(question)
    if not question_tokens:
        return 0.0

    best = 0.0
    for label in _TABLE_ROW_RE.findall(document):
        label_tokens = _keyword_tokens(label)
        if not label_tokens:
            continue
        overlap = len(question_tokens & label_tokens)
        if overlap:
            precision = overlap / len(label_tokens)
            best = max(best, overlap * 0.10 + precision * 0.24)
    return best


def _scope_bonus(question: str, document: str) -> float:
    question_norm = _normalize_text(question)
    document_norm = _normalize_text(document)

    bonus = 0.0
    has_quarter_markers = any(marker in document_norm for marker in ("q1 2025", "q2 2025", "q3 2025", "q4 2025"))
    has_year_series = bool(re.search(r"\b2021 2022 2023 2024 2025\b", document_norm))

    if "fy 2025" in question_norm:
        if has_year_series:
            bonus += 0.22
        if has_quarter_markers:
            bonus -= 0.12

    if "q4 2025" in question_norm:
        if "q4 2025" in document_norm or "4q 2025" in document_norm:
            bonus += 0.12
        if has_year_series and "q4 2025" not in document_norm and "4q 2025" not in document_norm:
            bonus -= 0.12

    if "q3 2025" in question_norm:
        if "q3 2025" in document_norm or "3q 2025" in document_norm:
            bonus += 0.12
        if has_year_series and "q3 2025" not in document_norm and "3q 2025" not in document_norm:
            bonus -= 0.12

    return bonus


def _lexical_rerank_bonus(question: str, document: str, metadata: dict[str, Any]) -> float:
    question_norm = _normalize_text(question)
    document_norm = _normalize_text(document)
    question_tokens = _keyword_tokens(question)
    document_tokens = _keyword_tokens(document)
    overlap = len(question_tokens & document_tokens)

    section = str(metadata.get("section", ""))
    section_tokens = _keyword_tokens(section)
    section_overlap = len(question_tokens & section_tokens)

    bonus = overlap * 0.015
    bonus += section_overlap * 0.08
    bonus += _table_row_bonus(question, document)
    bonus += _scope_bonus(question, document)

    if "q3 2025" in question_norm and "q3 2025" in document_norm:
        bonus += 0.04
    if "q4 2025" in question_norm and "q4 2025" in document_norm:
        bonus += 0.04

    return bonus


class EmbeddingFn(Protocol):
    def __call__(self, input: list[str]) -> list[list[float]]:
        ...


class VectorStore:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embedding_fn: EmbeddingFn | None = None,
    ) -> None:
        client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        if embedding_fn is None:
            embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        self._collection: Collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self) -> Collection:
        return self._collection

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source_file": c.source_file,
                    "page": c.page,
                    "section": c.section,
                }
                for c in chunks
            ],
        )
        return len(chunks)

    def count(self) -> int:
        return self._collection.count()

    def query(self, question: str, top_k: int = 4) -> dict:
        return self._collection.query(query_texts=[question], n_results=top_k)

    def get_all_chunks(self) -> dict[str, list[Any]]:
        data = self._collection.get(include=["documents", "metadatas"])
        return {
            "ids": data.get("ids", []),
            "documents": data.get("documents", []),
            "metadatas": data.get("metadatas", []),
        }

    def hybrid_query(self, question: str, top_k: int = 4, rrf_k: int = 60) -> dict:
        vector_top_k = max(top_k * 2, top_k)
        vector_response = self.query(question=question, top_k=vector_top_k)
        all_chunks = self.get_all_chunks()

        vector_ids = vector_response.get("ids", [[]])[0]
        vector_docs = vector_response.get("documents", [[]])[0]
        vector_metas = vector_response.get("metadatas", [[]])[0]
        vector_distances = vector_response.get("distances", [[]])[0]

        by_id: dict[str, dict[str, Any]] = {}
        vector_ranked_ids: list[str] = []
        for i, doc in enumerate(vector_docs):
            md = vector_metas[i] if i < len(vector_metas) else {}
            chunk_id = vector_ids[i] if i < len(vector_ids) else (
                f"vec::{md.get('source_file', 'unknown')}::{md.get('page', 0)}::{md.get('section', 'General')}::{i}"
            )
            vector_ranked_ids.append(chunk_id)
            by_id[chunk_id] = {
                "id": chunk_id,
                "document": doc,
                "metadata": md,
                "distance": vector_distances[i] if i < len(vector_distances) else None,
            }

        docs = all_chunks.get("documents", [])
        metas = all_chunks.get("metadatas", [])
        ids = all_chunks.get("ids", [])

        bm25_ranked_ids: list[str] = []
        if docs:
            tokenized_corpus = [d.lower().split() for d in docs]
            if any(tokenized_corpus):
                bm25 = BM25Okapi(tokenized_corpus)
                query_tokens = question.lower().split()
                scores = bm25.get_scores(query_tokens)
                scored_idx = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
                for idx in scored_idx:
                    if scores[idx] <= 0:
                        continue
                    doc_id = ids[idx] if idx < len(ids) else f"bm25::{idx}"
                    bm25_ranked_ids.append(doc_id)
                    if doc_id not in by_id:
                        by_id[doc_id] = {
                            "id": doc_id,
                            "document": docs[idx],
                            "metadata": metas[idx] if idx < len(metas) else {},
                            "distance": None,
                        }

        fused_scores: dict[str, float] = {}
        for rank, doc_id in enumerate(vector_ranked_ids, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
        for rank, doc_id in enumerate(bm25_ranked_ids, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.35 / (rrf_k + rank)

        for doc_id, entry in by_id.items():
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + _lexical_rerank_bonus(
                question=question,
                document=entry["document"],
                metadata=entry["metadata"],
            )

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        out_docs: list[str] = []
        out_metas: list[dict[str, Any]] = []
        out_distances: list[float | None] = []
        for doc_id, _ in ranked:
            entry = by_id[doc_id]
            out_docs.append(entry["document"])
            out_metas.append(entry["metadata"])
            out_distances.append(entry["distance"])

        return {
            "documents": [out_docs],
            "metadatas": [out_metas],
            "distances": [out_distances],
        }
