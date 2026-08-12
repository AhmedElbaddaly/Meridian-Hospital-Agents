"""
rag/build_index.py
--------------------
The ingestion pipeline: Documents -> Chunking -> Embeddings -> Vector
Database, exactly the shape from the lecture slide.

Run:
    python -m rag.build_index
"""

from __future__ import annotations

# --- allow running this file directly (python path/to/file.py), not
# --- just as a module (python -m pkg.file) -- both now work the same.
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)


from rag.chunking import chunk_corpus
from rag.embeddings import get_embedder
from rag.vector_store import VectorStore


def build(reset: bool = True) -> VectorStore:
    chunks = chunk_corpus()
    embedder = get_embedder(texts_to_fit_on=[c.text for c in chunks])

    store = VectorStore()
    if reset:
        store.reset()

    for chunk in chunks:
        vector = embedder.embed(chunk.text)
        store.upsert(
            chunk.chunk_id, chunk.text, vector,
            doc_id=chunk.doc_id, protocol_id=chunk.protocol_id,
            department=chunk.department, last_reviewed=chunk.last_reviewed,
            title=chunk.title,
        )

    print(f"Indexed {len(chunks)} chunks from {len(set(c.doc_id for c in chunks))} documents.")
    return store


if __name__ == "__main__":
    build()
