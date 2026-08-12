"""
rag/embeddings.py
-------------------
Produces dense vectors for chunks and queries.

This repo has no network access for a hosted embedding model in this
environment, so the default path is a small, deterministic, fully offline
embedder: TF-IDF over the corpus vocabulary, dimensionality-reduced with
truncated SVD (i.e. classic LSA) to a fixed-size dense vector. This is a
legitimate embedding technique (semantically-similar chunks do end up
closer in the reduced space), it is 100% reproducible without any
credentials, and it is a drop-in stand-in for a hosted embedding call.

To use a real embedding provider instead (e.g. Voyage AI, OpenAI,
Anthropic-recommended providers), implement `RealEmbedder` below following
the same `.fit(texts)` / `.embed(text)` interface and swap it in
`get_embedder()`. Nothing else in `rag/` needs to change -- vector_store.py
only depends on this interface.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

EMBED_DIM = 48

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDER_CACHE_PATH = os.path.join(BASE_DIR, "_embedder.pkl")


class OfflineEmbedder:
    """TF-IDF -> TruncatedSVD dense embedding. Deterministic, offline."""

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.svd: TruncatedSVD | None = None
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        tfidf = self.vectorizer.fit_transform(texts)
        n_components = min(self.dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        n_components = max(n_components, 2)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(tfidf)
        self._fitted = True

    def embed(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("OfflineEmbedder must be fit() before embed()")
        tfidf = self.vectorizer.transform([text])
        vec = self.svd.transform(tfidf)[0]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])

    def save(self, path: str = EMBEDDER_CACHE_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str = EMBEDDER_CACHE_PATH) -> "OfflineEmbedder":
        with open(path, "rb") as f:
            return pickle.load(f)


_embedder_singleton: OfflineEmbedder | None = None


def get_embedder(texts_to_fit_on: list[str] | None = None) -> OfflineEmbedder:
    """
    Returns a process-wide embedder fitted once on the corpus vocabulary.
    Call with `texts_to_fit_on` the first time (e.g. from
    vector_store.build_index()); later callers can omit it and reuse the
    fitted singleton.
    """
    global _embedder_singleton
    if _embedder_singleton is None:
        if texts_to_fit_on is None:
            raise RuntimeError("Embedder not yet fitted; pass texts_to_fit_on once at index build time.")
        _embedder_singleton = OfflineEmbedder()
        _embedder_singleton.fit(texts_to_fit_on)
    return _embedder_singleton
