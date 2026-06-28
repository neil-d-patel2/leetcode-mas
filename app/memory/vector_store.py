"""A tiny in-memory vector store with cosine-similarity search.

This is deliberately minimal — a list of (vector, payload) pairs and a
linear scan at query time. For a personal LeetCode library (hundreds of
files) that is plenty fast, and it keeps the concept transparent. Swap in
FAISS / Chroma later behind the same `add` / `query` interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _cosine(a: list[float], b: list[float]) -> float:
    # Vectors from our embedders are already L2-normalized, so the dot
    # product IS the cosine similarity.
    return sum(x * y for x, y in zip(a, b))


@dataclass
class _Record:
    vector: list[float]
    payload: dict[str, Any]


@dataclass
class VectorStore:
    """Holds embedded documents and finds the most similar ones."""

    embedder: Any  # anything with .embed(list[str]) -> list[list[float]]
    _records: list[_Record] = field(default_factory=list)

    def add_documents(self, texts: list[str], payloads: list[dict[str, Any]]) -> None:
        """Embed `texts` and store them alongside their metadata payloads."""
        if not texts:
            return
        for vec, payload in zip(self.embedder.embed(texts), payloads):
            self._records.append(_Record(vector=vec, payload=payload))

    def query(self, text: str, top_k: int = 3) -> list[tuple[float, dict[str, Any]]]:
        """Return up to `top_k` (score, payload) pairs, most similar first."""
        if not self._records:
            return []
        qvec = self.embedder.embed([text])[0]
        scored = [(_cosine(qvec, r.vector), r.payload) for r in self._records]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._records)
