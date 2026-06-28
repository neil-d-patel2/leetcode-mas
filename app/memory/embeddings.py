"""Turn text into vectors so we can measure similarity between problems.

Two backends:
  - `HashingEmbedder` (default): a dependency-free bag-of-words hashing
    embedding. It is crude but real enough to demonstrate retrieval, and
    it runs anywhere with just the standard library.
  - `OllamaEmbedder`: calls Ollama's /api/embeddings with a real
    embedding model (config.embed_model). Use this once Ollama is set up
    for much better similarity.

Both expose the same `.embed(texts) -> list[list[float]]` interface, so
the vector store doesn't care which one it's given.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from app.config import Config

_TOKEN_RE = re.compile(r"[a-zA-Z_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


class HashingEmbedder:
    """Maps text to a fixed-size, L2-normalized bag-of-words vector.

    Each token is hashed into one of `dim` buckets; counts go in those
    buckets. Similar texts share tokens, so their vectors point in
    similar directions (high cosine similarity). No training, no deps.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in _tokenize(text):
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            vectors.append(_normalize(vec))
        return vectors


class OllamaEmbedder:
    """Real embeddings via Ollama. Requires `requests` and a pulled model."""

    def __init__(self, config: "Config") -> None:
        self.config = config

    def embed(self, texts: list[str]) -> list[list[float]]:
        import requests  # lazy import; only needed for real embeddings

        url = f"{self.config.ollama_host}/api/embeddings"
        vectors = []
        for text in texts:
            resp = requests.post(
                url,
                json={"model": self.config.embed_model, "prompt": text},
                timeout=120,
            )
            resp.raise_for_status()
            vectors.append(_normalize(resp.json()["embedding"]))
        return vectors


def build_embedder(config: "Config"):
    """Pick an embedder based on config (stub mode -> hashing)."""
    if config.use_stub_llm:
        return HashingEmbedder()
    return OllamaEmbedder(config)
