"""
rag/naive_rag.py
------------------
The baseline pipeline: embed the query, run a single unfiltered ANN search
against the vector store, generate from whatever comes back. No keyword
matching, no multi-hop, no filtering. This is deliberately the "does the
simplest thing work" baseline the other two architectures are compared
against.
"""

from __future__ import annotations

# --- allow running this file directly (python path/to/file.py), not
# --- just as a module (python -m pkg.file) -- both now work the same.
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)


from rag.embeddings import get_embedder
from rag.generation import generate_answer
from rag.vector_store import RetrievedChunk, VectorStore


def naive_rag_answer(question: str, store: VectorStore, top_k: int = 3) -> tuple[str, list[RetrievedChunk]]:
    embedder = get_embedder()
    query_vec = embedder.embed(question)
    chunks = store.query(query_vec, top_k=top_k)
    answer = generate_answer(question, chunks)
    return answer, chunks


if __name__ == "__main__":
    from rag.build_index import build

    store = build()
    q = "What's the standard fasting window before sedation?"
    answer, chunks = naive_rag_answer(q, store)
    print("Q:", q)
    print("Retrieved:", [c.protocol_id for c in chunks])
    print("A:", answer)
