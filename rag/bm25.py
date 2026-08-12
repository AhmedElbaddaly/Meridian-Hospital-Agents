"""
rag/bm25.py
------------
A small, from-scratch BM25 keyword scorer -- `rank_bm25` (the resource the
lab points to) isn't installable in this offline sandbox, so this
implements the same Okapi BM25 formula directly. Swap in `rank_bm25.BM25Okapi`
verbatim if network access is available; the interface below matches it
closely on purpose (`.get_scores(query_tokens) -> list[float]`).

BM25 is what catches exact identifiers -- "4.2b", "MRSA" -- that a
dense/semantic embedding tends to blur together with similar-but-wrong
chunks. That is precisely why rag/hybrid_search.py fuses this with vector
similarity instead of relying on either alone.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9\.]+", text.lower())


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = []
        self.doc_freqs: list[Counter] = []
        self.df: Counter = Counter()
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0
        self.N = 0

    def fit(self, documents: list[str]) -> None:
        self.doc_tokens = [tokenize(d) for d in documents]
        self.N = len(self.doc_tokens)
        self.doc_freqs = [Counter(toks) for toks in self.doc_tokens]

        for freqs in self.doc_freqs:
            for term in freqs:
                self.df[term] += 1

        self.idf = {
            term: math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))
            for term, freq in self.df.items()
        }
        self.avgdl = sum(len(t) for t in self.doc_tokens) / max(self.N, 1)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.N
        for i, freqs in enumerate(self.doc_freqs):
            dl = len(self.doc_tokens[i])
            for term in query_tokens:
                if term not in freqs:
                    continue
                f = freqs[term]
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[i] += idf * (f * (self.k1 + 1)) / max(denom, 1e-9)
        return scores
