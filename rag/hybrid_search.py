"""
rag/hybrid_search.py
-----------------------
Vector similarity (semantic) + BM25 (exact keyword/identifier) fused into
one ranking. This is what naive RAG structurally cannot do: a citation like
"4.2b" or a drug name doesn't embed distinctively (it looks like noise to a
dense embedding trained on general similarity), but it is exactly what BM25
is built to score highly, since it is a literal token.

Fusion strategy: reciprocal rank fusion (RRF) -- more robust than a raw
weighted-score average since it doesn't require the two score scales
(cosine similarity vs. BM25 score) to be comparable.
"""

from __future__ import annotations

# --- allow running this file directly (python path/to/file.py), not
# --- just as a module (python -m pkg.file) -- both now work the same.
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)


from dataclasses import dataclass

from rag.bm25 import BM25, tokenize
from rag.chunking import chunk_corpus
from rag.embeddings import get_embedder
from rag.generation import generate_answer
from rag.vector_store import RetrievedChunk, VectorStore

RRF_K = 60  # standard reciprocal-rank-fusion constant


@dataclass
class HybridIndex:
    bm25: BM25
    chunk_ids_in_order: list[str]


def build_hybrid_index() -> HybridIndex:
    chunks = chunk_corpus()
    bm25 = BM25()
    bm25.fit([c.text for c in chunks])
    return HybridIndex(bm25=bm25, chunk_ids_in_order=[c.chunk_id for c in chunks])


def hybrid_search(question: str, store: VectorStore, hybrid_index: HybridIndex,
                   top_k: int = 3, vector_pool: int = 8) -> list[RetrievedChunk]:
    embedder = get_embedder()
    query_vec = embedder.embed(question)

    # semantic side: wider pool than top_k, since fusion needs ranks not just winners
    vector_hits = store.query(query_vec, top_k=vector_pool)
    vector_rank = {c.chunk_id: rank for rank, c in enumerate(vector_hits)}

    # keyword side
    scores = hybrid_index.bm25.get_scores(tokenize(question))
    bm25_ranked = sorted(
        zip(hybrid_index.chunk_ids_in_order, scores), key=lambda x: -x[1]
    )
    bm25_rank = {cid: rank for rank, (cid, score) in enumerate(bm25_ranked) if score > 0}

    all_ids = set(vector_rank) | set(bm25_rank)
    fused = []
    for cid in all_ids:
        rrf_score = 0.0
        if cid in vector_rank:
            rrf_score += 1.0 / (RRF_K + vector_rank[cid])
        if cid in bm25_rank:
            rrf_score += 1.0 / (RRF_K + bm25_rank[cid])
        fused.append((cid, rrf_score))
    fused.sort(key=lambda x: -x[1])

    top_ids = [cid for cid, _ in fused[:top_k]]
    by_id = {c.chunk_id: c for c in store.all_chunks()}
    results = []
    for cid in top_ids:
        if cid in by_id:
            c = by_id[cid]
            results.append(RetrievedChunk(
                chunk_id=c.chunk_id, text=c.text, protocol_id=c.protocol_id,
                department=c.department, title=c.title,
                score=dict(fused)[cid],
            ))
    return results


def hybrid_rag_answer(question: str, store: VectorStore, hybrid_index: HybridIndex,
                       top_k: int = 3) -> tuple[str, list[RetrievedChunk]]:
    chunks = hybrid_search(question, store, hybrid_index, top_k=top_k)
    answer = generate_answer(question, chunks)
    return answer, chunks


if __name__ == "__main__":
    from rag.build_index import build

    store = build()
    hidx = build_hybrid_index()
    q = "What does Protocol 4.2b say about cardiac-risk patients?"
    answer, chunks = hybrid_rag_answer(q, store, hidx)
    print("Q:", q)
    print("Retrieved:", [c.protocol_id for c in chunks])
    print("A:", answer)
