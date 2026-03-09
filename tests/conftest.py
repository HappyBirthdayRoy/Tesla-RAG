from __future__ import annotations

from collections.abc import Iterator

import pytest


class FakeEmbeddingFunction:
    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in input:
            v = [0.0] * 32
            for i, ch in enumerate(text.encode("utf-8")):
                v[i % 32] += (ch % 31) / 31.0
            vectors.append(v)
        return vectors


@pytest.fixture
def fake_embedding() -> Iterator[FakeEmbeddingFunction]:
    yield FakeEmbeddingFunction()
