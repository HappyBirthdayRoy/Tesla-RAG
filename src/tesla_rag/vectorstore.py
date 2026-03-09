from __future__ import annotations

from dataclasses import asdict
from typing import Protocol

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from tesla_rag.models import Chunk


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
